"""Sanity-check inspector for MVP1 graph tensors before model training."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 data flow та anti-blind coding)
# - DATA_FLOWS_MVP1.MD -> шлях RawTrace -> PrefixSlice -> GraphTensorContract
# - LLD_MVP1.MD -> розділ 4-5 (фічі та графові тензори)

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Sequence

import torch
from torch_geometric.loader import DataLoader

from src.cli import load_yaml_config, prepare_data, set_seed


def _decode_first_nodes(
    batch,
    reverse_activity_vocab,
    reverse_resource_vocab,
    ordered_extra_keys: Sequence[str],
    extra_vocabs: Dict[str, Dict[str, int]],
    max_nodes: int = 5,
) -> None:
    """Decode selected graph nodes from split tensors (x_cat + x_num)."""
    x_cat = batch.x_cat
    x_num = batch.x_num

    if hasattr(batch, "batch") and batch.batch.numel() == x_cat.size(0):
        node_indices = (batch.batch == 0).nonzero(as_tuple=False).view(-1)
    else:
        # Fallback for malformed/empty batch vector.
        node_indices = torch.arange(x_cat.size(0), dtype=torch.long, device=x_cat.device)

    if node_indices.numel() == 0 and x_cat.size(0) > 0:
        node_indices = torch.arange(x_cat.size(0), dtype=torch.long, device=x_cat.device)

    print("\n=== First graph dynamic node decode ===")
    if node_indices.numel() == 0:
        print("No nodes found in current batch.")
        return

    categorical_keys = [key for key in ordered_extra_keys if key in extra_vocabs]
    scalar_extra_keys = [key for key in ordered_extra_keys if key not in extra_vocabs]

    selected_nodes = node_indices[:max_nodes].tolist()
    last_node = int(node_indices[-1].item())
    if last_node not in selected_nodes:
        selected_nodes.append(last_node)

    cat_key_to_offset = {key: idx for idx, key in enumerate(categorical_keys)}

    for order, node_idx in enumerate(selected_nodes, start=1):
        cat_row = x_cat[node_idx]
        num_row = x_num[node_idx]

        act_idx = int(cat_row[0].item()) if cat_row.numel() > 0 else 0
        res_idx = int(cat_row[1].item()) if cat_row.numel() > 1 else 0
        act_name = reverse_activity_vocab.get(act_idx, "<UNK>")
        res_name = reverse_resource_vocab.get(res_idx, "<UNK>")

        duration = float(num_row[0].item()) if num_row.numel() > 0 else 0.0
        t_case = float(num_row[1].item()) if num_row.numel() > 1 else 0.0
        t_prev = float(num_row[2].item()) if num_row.numel() > 2 else 0.0

        print(
            f"node#{order} (global_idx={node_idx}) | "
            f"activity_idx={act_idx} -> {act_name} | "
            f"resource_idx={res_idx} -> {res_name} | "
            f"base_time=[duration_z={duration:.4f}, time_since_case_start_z={t_case:.4f}, "
            f"time_since_prev_event_z={t_prev:.4f}]"
        )

        if scalar_extra_keys:
            scalar_preview: List[str] = []
            for idx, key in enumerate(scalar_extra_keys):
                pos = 3 + idx
                if pos >= num_row.numel():
                    continue
                val = float(num_row[pos].item())
                if abs(val) > 1e-8:
                    scalar_preview.append(f"{key}={val:.4f}")
            print(f"  extra_scalars: {', '.join(scalar_preview[:8]) if scalar_preview else '<all-zero>'}")

        if categorical_keys:
            decoded_categories: List[str] = []
            for key in categorical_keys:
                col = 2 + cat_key_to_offset[key]
                if col < cat_row.numel():
                    local_idx = int(cat_row[col].item())
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
    """Print split feature structure and consistency checks for FeatureLayout."""
    feature_layout = prepared["feature_layout"]
    cat_feature_count = len(feature_layout.cat_feature_names)
    num_dim = int(feature_layout.num_dim)

    print("\n=== Feature Breakdown ===")
    print(f"Categorical Features (count: {cat_feature_count})")
    print(f"Numerical/Encoded Channels (count: {num_dim})")

    calculated_total = cat_feature_count + num_dim
    layout_total = sum(1 for _ in feature_layout.cat_feature_names) + int(feature_layout.num_dim)
    builder_total = int(prepared["graph_builder"].input_dim)

    print(f"Total logical channels: {calculated_total}")
    print(f"FeatureLayout total: {layout_total}")
    print(f"GraphBuilder declared total: {builder_total}")
    print(
        "Consistency check (cat_count + num_dim == FeatureLayout == GraphBuilder): "
        f"{calculated_total == layout_total == builder_total}"
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
    print(f"x_cat: {tuple(batch.x_cat.shape)}")
    print(f"x_num: {tuple(batch.x_num.shape)}")
    print(f"edge_index: {tuple(batch.edge_index.shape)}")
    print(f"y: {tuple(batch.y.shape)}")
    print(f"batch: {tuple(batch.batch.shape)}")

    has_nan = bool(torch.isnan(batch.x_num).any().item())
    is_all_zero = bool(torch.all(batch.x_num == 0).item())
    has_non_zero = bool(torch.any(batch.x_num != 0).item())

    print("\n=== Sanity Checks ===")
    print(f"x_num contains NaN: {has_nan}")
    print(f"x_num is all zeros: {is_all_zero}")
    print(f"x_num has non-zero features: {has_non_zero}")

    logger.info("Train dataset graphs: %d | train batches: %d", len(train_dataset), len(train_loader))

    _print_vocab_summary(prepared)
    _print_feature_breakdown(prepared)

    _decode_first_nodes(
        batch=batch,
        reverse_activity_vocab=prepared["reverse_activity_vocab"],
        reverse_resource_vocab=prepared["reverse_resource_vocab"],
        ordered_extra_keys=prepared.get("ordered_extra_keys", []),
        extra_vocabs=prepared.get("extra_vocabs", {}),
        max_nodes=5,
    )


if __name__ == "__main__":
    main()
