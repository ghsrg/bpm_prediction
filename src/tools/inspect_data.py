"""Sanity-check inspector for MVP1 graph tensors before model training."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 data flow та anti-blind coding)
# - DATA_FLOWS_MVP1.MD -> шлях RawTrace -> PrefixSlice -> GraphTensorContract
# - LLD_MVP1.MD -> розділ 4-5 (фічі та графові тензори)

from __future__ import annotations

import argparse
import logging

import torch
from torch_geometric.loader import DataLoader

from src.cli import load_yaml_config, prepare_data, set_seed


def _decode_first_nodes(
    batch,
    reverse_activity_vocab,
    reverse_resource_vocab,
    num_activity_features: int,
    num_resource_features: int,
    max_nodes: int = 5,
) -> None:
    """Decode first graph nodes from one-hot segments back to readable labels."""
    x = batch.x
    node_indices = (batch.batch == 0).nonzero(as_tuple=False).view(-1)

    print("\n=== First graph node decode (up to 5 nodes) ===")
    if node_indices.numel() == 0:
        print("No nodes found for first graph in batch.")
        return

    for order, node_idx in enumerate(node_indices[:max_nodes].tolist(), start=1):
        row = x[node_idx]
        act_slice = row[:num_activity_features]
        res_slice = row[num_activity_features : num_activity_features + num_resource_features]
        num_slice = row[num_activity_features + num_resource_features :]

        act_idx = int(torch.argmax(act_slice).item())
        res_idx = int(torch.argmax(res_slice).item())

        act_name = reverse_activity_vocab.get(act_idx, "<UNK>")
        res_name = reverse_resource_vocab.get(res_idx, "<UNK>")

        print(
            f"node#{order} (global_idx={node_idx}) | "
            f"activity_idx={act_idx} -> {act_name} (onehot={act_slice[act_idx].item():.1f}) | "
            f"resource_idx={res_idx} -> {res_name} (onehot={res_slice[res_idx].item():.1f}) | "
            f"numeric=[duration_z={num_slice[0].item():.4f}, "
            f"time_since_case_start_z={num_slice[1].item():.4f}, "
            f"time_since_prev_event_z={num_slice[2].item():.4f}]"
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

    _decode_first_nodes(
        batch=batch,
        reverse_activity_vocab=prepared["reverse_activity_vocab"],
        reverse_resource_vocab=prepared["reverse_resource_vocab"],
        num_activity_features=len(prepared["activity_vocab"]),
        num_resource_features=len(prepared["resource_vocab"]),
        max_nodes=5,
    )


if __name__ == "__main__":
    main()
