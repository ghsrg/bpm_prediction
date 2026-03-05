"""Sanity-check inspector for MVP1 graph tensors before model training."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 data flow та anti-blind coding)
# - DATA_FLOWS_MVP1.MD -> шлях RawTrace -> PrefixSlice -> GraphTensorContract
# - LLD_MVP1.MD -> розділ 4-5 (фічі та графові тензори)

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Sequence, Tuple

import torch
from torch_geometric.loader import DataLoader

from src.cli import load_yaml_config, prepare_data, set_seed


def _decode_first_nodes(
    batch,
    reverse_activity_vocab,
    reverse_resource_vocab,
    num_activity_features: int,
    num_resource_features: int,
    ordered_extra_keys: Sequence[str],
    extra_vocabs: Dict[str, Dict[str, int]],
    max_nodes: int = 5,
) -> None:
    """Decode selected graph nodes from dynamic feature slices back to readable labels."""
    x = batch.x
    node_indices = (batch.batch == 0).nonzero(as_tuple=False).view(-1)

    print("\n=== First graph dynamic node decode ===")
    if node_indices.numel() == 0:
        print("No nodes found for first graph in batch.")
        return

    scalar_extra_keys = [key for key in ordered_extra_keys if key not in extra_vocabs]

    categorical_offsets: List[Tuple[str, int, int]] = []
    cat_start = num_activity_features + num_resource_features + 3 + len(scalar_extra_keys)
    running = cat_start
    for key in ordered_extra_keys:
        vocab = extra_vocabs.get(key)
        if vocab is None:
            continue
        width = len(vocab)
        categorical_offsets.append((key, running, running + width))
        running += width

    selected_nodes = node_indices[:max_nodes].tolist()
    last_node = int(node_indices[-1].item())
    if last_node not in selected_nodes:
        selected_nodes.append(last_node)

    for order, node_idx in enumerate(selected_nodes, start=1):
        row = x[node_idx]
        act_slice = row[:num_activity_features]
        res_slice = row[num_activity_features : num_activity_features + num_resource_features]

        base_time_start = num_activity_features + num_resource_features
        time_slice = row[base_time_start : base_time_start + 3]
        scalar_start = base_time_start + 3
        scalar_end = scalar_start + len(scalar_extra_keys)
        scalar_slice = row[scalar_start:scalar_end]

        act_idx = int(torch.argmax(act_slice).item())
        res_idx = int(torch.argmax(res_slice).item())

        act_name = reverse_activity_vocab.get(act_idx, "<UNK>")
        res_name = reverse_resource_vocab.get(res_idx, "<UNK>")

        print(
            f"node#{order} (global_idx={node_idx}) | "
            f"activity_idx={act_idx} -> {act_name} (onehot={act_slice[act_idx].item():.1f}) | "
            f"resource_idx={res_idx} -> {res_name} (onehot={res_slice[res_idx].item():.1f}) | "
            f"base_time=[duration_z={time_slice[0].item():.4f}, "
            f"time_since_case_start_z={time_slice[1].item():.4f}, "
            f"time_since_prev_event_z={time_slice[2].item():.4f}]"
        )

        if scalar_extra_keys:
            scalar_preview = []
            for idx, key in enumerate(scalar_extra_keys):
                val = float(scalar_slice[idx].item())
                if abs(val) > 1e-8:
                    scalar_preview.append(f"{key}={val:.4f}")
            if not scalar_preview:
                scalar_preview = [f"{key}={float(scalar_slice[idx].item()):.4f}" for idx, key in enumerate(scalar_extra_keys[:3])]
            print(f"  extra_scalars: {', '.join(scalar_preview[:8])}")

        if categorical_offsets:
            decoded_categories: List[str] = []
            for key, start_idx, end_idx in categorical_offsets:
                cat_slice = row[start_idx:end_idx]
                active = (cat_slice == 1.0).nonzero(as_tuple=False).view(-1)
                if active.numel() > 0:
                    local_idx = int(active[0].item())
                    reverse_vocab = {idx: val for val, idx in extra_vocabs[key].items()}
                    decoded_val = reverse_vocab.get(local_idx, "<UNK>")
                    decoded_categories.append(f"{key}={decoded_val}")
                else:
                    decoded_categories.append(f"{key}=<all-zero>")
            print(f"  extra_categorical: {', '.join(decoded_categories[:8])}")


def _print_vocab_summary(prepared: Dict[str, object]) -> None:
    """Print vocabulary cardinalities for baseline and extra categorical features."""
    activity_vocab = prepared["activity_vocab"]
    resource_vocab = prepared["resource_vocab"]
    extra_vocabs = prepared.get("extra_vocabs", {})

    print("\n=== Vocabularies Summary ===")
    print(f"activity_vocab: {len(activity_vocab)}")
    print(f"resource_vocab: {len(resource_vocab)}")

    if isinstance(extra_vocabs, dict) and extra_vocabs:
        print("extra_vocabs (sorted by size desc):")
        sorted_items = sorted(extra_vocabs.items(), key=lambda item: len(item[1]), reverse=True)
        for key, vocab in sorted_items:
            print(f"  - {key}: {len(vocab)}")
    else:
        print("extra_vocabs: <none>")


def _print_feature_breakdown(prepared: Dict[str, object]) -> None:
    """Print detailed input_dim decomposition and consistency checks."""
    num_activity = len(prepared["activity_vocab"])
    num_resource = len(prepared["resource_vocab"])
    base_time = 3
    ordered_extra_keys = prepared.get("ordered_extra_keys", [])
    extra_vocabs = prepared.get("extra_vocabs", {})

    scalar_extra_keys = [key for key in ordered_extra_keys if key not in extra_vocabs]
    scalar_extra = len(scalar_extra_keys)

    categorical_parts = [(key, len(extra_vocabs[key])) for key in ordered_extra_keys if key in extra_vocabs]
    categorical_total = sum(width for _, width in categorical_parts)

    calculated_dim = num_activity + num_resource + base_time + scalar_extra + categorical_total
    prepared_dim = int(prepared.get("input_dim", -1))
    builder_dim = int(prepared["graph_builder"].input_dim)

    breakdown_parts = [
        f"Activity ({num_activity})",
        f"Resource ({num_resource})",
        f"Base Time ({base_time})",
        f"Extra Scalars ({scalar_extra})",
    ]
    breakdown_parts.extend([f"{name} ({size})" for name, size in categorical_parts])

    print("\n=== Feature Breakdown ===")
    print(f"Input Dimension ({prepared_dim}) = " + " + ".join(breakdown_parts))
    print(f"Calculated input_dim: {calculated_dim}")
    print(f"GraphBuilder input_dim: {builder_dim}")
    print(
        "Consistency check (calculated == prepared == graph_builder): "
        f"{calculated_dim == prepared_dim == builder_dim}"
    )


def main() -> None:
    """Load config, build train batch, and print tensor sanity diagnostics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Inspect one training batch for tensor sanity checks.")
    parser.add_argument("--config", default="configs/permit_log.yaml", help="YAML experiment config path or filename.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    set_seed(int(config.get("seed", 42)))

    prepared = prepare_data(config)
    train_dataset = prepared["train_dataset"]

    if not train_dataset:
        print("Train dataset is empty. Check data split and source log.")
        return

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 128))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    batch = next(iter(train_loader))

    print("=== Tensor Shapes ===")
    print(f"x: {tuple(batch.x.shape)}")
    print(f"edge_index: {tuple(batch.edge_index.shape)}")
    print(f"y: {tuple(batch.y.shape)}")
    print(f"batch: {tuple(batch.batch.shape)}")

    has_nan = bool(torch.isnan(batch.x).any().item())
    is_all_zero = bool(torch.all(batch.x == 0).item())
    has_non_zero = bool(torch.any(batch.x != 0).item())

    print("\n=== Sanity Checks ===")
    print(f"x contains NaN: {has_nan}")
    print(f"x is all zeros: {is_all_zero}")
    print(f"x has non-zero features: {has_non_zero}")

    logger.info("Train dataset graphs: %d | train batches: %d", len(train_dataset), len(train_loader))

    _print_vocab_summary(prepared)
    _print_feature_breakdown(prepared)

    _decode_first_nodes(
        batch=batch,
        reverse_activity_vocab=prepared["reverse_activity_vocab"],
        reverse_resource_vocab=prepared["reverse_resource_vocab"],
        num_activity_features=len(prepared["activity_vocab"]),
        num_resource_features=len(prepared["resource_vocab"]),
        ordered_extra_keys=prepared.get("ordered_extra_keys", []),
        extra_vocabs=prepared.get("extra_vocabs", {}),
        max_nodes=5,
    )


if __name__ == "__main__":
    main()
