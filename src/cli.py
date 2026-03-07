"""Composition Root for MVP1 training pipeline."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (Clean Architecture, MVP1 scope) і розділ 5 (послідовність потоків)
# - ARCHITECTURE_RULES.md -> розділ 2-4 (Application orchestration через порти)
# - DATA_FLOWS_MVP1.MD -> розділ 3.1 (Training Pipeline)
# - EVF_MVP1.MD -> розділ 3 (Strict Temporal Split) і розділ 5 (MLflow tracking)

from __future__ import annotations

import argparse
import math
from pathlib import Path
import logging
import random
from typing import Any, Dict, List, Sequence, Tuple, Type

import numpy as np
import torch
from torch_geometric.data import Data

from src.adapters.ingestion.xes_adapter import XESAdapter
from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.raw_trace import RawTrace
from src.domain.entities.feature_config import parse_feature_configs
from src.domain.models.base_gnn import BaseGNN
from src.domain.models.baseline_gat import BaselineGATv2
from src.domain.models.baseline_gcn import BaselineGCN
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy
from src.infrastructure.config.yaml_loader import load_yaml_with_includes
from src.infrastructure.tracking.mlflow_tracker import MLflowTracker


def _resolve_config_path(config_arg: str) -> Path:
    """Resolve config path; if filename only, lookup in ./configs."""
    path = Path(config_arg)
    if path.exists():
        return path

    fallback = Path("configs") / config_arg
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Config file not found: {config_arg}")


def load_yaml_config(config_arg: str) -> Dict[str, Any]:
    """Load runtime configuration from YAML file with recursive include support."""
    config_path = _resolve_config_path(config_arg)

    # Завантажуємо конфіг із підтримкою include/deep-merge для модульних playbook-ів.
    loaded = load_yaml_with_includes(config_path)

    # Нормалізуємо секцію experiment для майбутнього роутингу режимів (train/eval/infer).
    experiment_cfg = loaded.get("experiment")
    if experiment_cfg is None:
        loaded["experiment"] = {"mode": "train"}
    elif isinstance(experiment_cfg, dict):
        experiment_cfg.setdefault("mode", "train")
    else:
        raise ValueError("Config key 'experiment' must be a mapping if provided.")

    return loaded


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across random/numpy/torch backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _extract_base_vocabularies(feature_encoder: FeatureEncoder) -> Tuple[Dict[str, int], Dict[str, int], str]:
    """Extract activity/resource vocabularies from feature encoder artifacts."""
    activity_feature = feature_encoder.activity_feature_name
    resource_feature = feature_encoder.role_to_feature.get("resource", "org:resource")
    activity_vocab = feature_encoder.categorical_vocabs.get(activity_feature, {"<UNK>": 0})
    resource_vocab = feature_encoder.categorical_vocabs.get(resource_feature, {"<UNK>": 0})
    return activity_vocab, resource_vocab, activity_feature


def _build_normalization_stats() -> Dict[str, Dict[str, float]]:
    """Create MVP1 placeholder normalization stats (mu=0, sigma=1)."""
    return {
        "duration": {"mu": 0.0, "sigma": 1.0},
        "time_since_case_start": {"mu": 0.0, "sigma": 1.0},
        "time_since_previous_event": {"mu": 0.0, "sigma": 1.0},
    }


def _parse_split_ratio(data_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """Parse split ratio with strict validation for train/val/test proportions."""
    raw_ratio = data_cfg.get("split_ratio", [0.7, 0.2, 0.1])
    if not isinstance(raw_ratio, Sequence) or isinstance(raw_ratio, (str, bytes)) or len(raw_ratio) != 3:
        raise ValueError("data.split_ratio must be a 3-item list [train, val, test].")

    ratios = [float(item) for item in raw_ratio]
    if any(item < 0.0 for item in ratios):
        raise ValueError("data.split_ratio must contain non-negative values.")

    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"data.split_ratio must sum to 1.0, got {ratio_sum:.6f}")

    return ratios[0], ratios[1], ratios[2]


def _apply_fraction(traces: Sequence[RawTrace], fraction: float) -> List[RawTrace]:
    """Apply chronological fraction on traces after temporal ordering."""
    traces_with_events = [trace for trace in traces if trace.events]
    ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)

    if fraction >= 1.0:
        return ordered
    if fraction <= 0.0:
        raise ValueError("data.fraction must be within (0.0, 1.0].")

    keep_count = max(1, int(len(ordered) * fraction))
    return ordered[:keep_count]


def _strict_temporal_split(
    traces: Sequence[RawTrace],
    split_ratio: Tuple[float, float, float],
    split_strategy: str,
) -> Tuple[List[RawTrace], List[RawTrace], List[RawTrace]]:
    """Apply strict chronological split by configured strategy and ratio."""
    if split_strategy != "time":
        raise ValueError(f"Unsupported data.split_strategy '{split_strategy}'. Only 'time' is allowed for MVP1.")

    train_ratio, val_ratio, _ = split_ratio
    traces_with_events = [trace for trace in traces if trace.events]
    ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)

    total = len(ordered)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return list(ordered[:train_end]), list(ordered[train_end:val_end]), list(ordered[val_end:])


def _build_graph_dataset(
    traces: Sequence[RawTrace],
    prefix_policy: PrefixPolicy,
    graph_builder: BaselineGraphBuilder,
) -> List[Data]:
    """Convert traces into a list of PyG Data graphs via prefix slicing + graph builder."""
    dataset: List[Data] = []
    for trace in traces:
        for prefix_slice in prefix_policy.generate_slices(trace):
            contract = graph_builder.build_graph(prefix_slice)
            dataset.append(
                Data(
                    x_cat=contract["x_cat"],
                    x_num=contract["x_num"],
                    edge_index=contract["edge_index"],
                    edge_type=contract["edge_type"],
                    y=contract["y"],
                    num_nodes=int(contract["num_nodes"]),
                )
            )
    return dataset


def prepare_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare shared data artifacts for CLI and inspector without logic duplication."""
    data_cfg = config.get("data", {})
    mapping_cfg = config.get("mapping", {})

    log_path = str(data_cfg.get("log_path", ""))
    if not log_path:
        raise ValueError("Config must define data.log_path")

    fraction = float(data_cfg.get("fraction", 1.0))
    split_strategy = str(data_cfg.get("split_strategy", "time")).strip().lower()
    split_ratio = _parse_split_ratio(data_cfg)

    xes_adapter = XESAdapter()
    prefix_policy = PrefixPolicy()

    traces = list(xes_adapter.read(log_path, mapping_cfg))
    traces = _apply_fraction(traces, fraction)
    feature_configs = parse_feature_configs(config)
    feature_encoder = FeatureEncoder(feature_configs=feature_configs, traces=traces)
    feature_layout = feature_encoder.feature_layout

    activity_vocab, resource_vocab, activity_feature = _extract_base_vocabularies(feature_encoder)
    reverse_activity_vocab = {idx: key for key, idx in activity_vocab.items()}
    reverse_resource_vocab = {idx: key for key, idx in resource_vocab.items()}

    normalization_stats = _build_normalization_stats()
    graph_builder = BaselineGraphBuilder(feature_encoder=feature_encoder)
    train_traces, val_traces, test_traces = _strict_temporal_split(traces, split_ratio, split_strategy)

    train_dataset = _build_graph_dataset(train_traces, prefix_policy, graph_builder)
    val_dataset = _build_graph_dataset(val_traces, prefix_policy, graph_builder)
    test_dataset = _build_graph_dataset(test_traces, prefix_policy, graph_builder)

    return {
        "log_path": log_path,
        "mapping_config": mapping_cfg,
        "data_config": {
            "fraction": fraction,
            "split_strategy": split_strategy,
            "split_ratio": list(split_ratio),
        },
        "activity_vocab": activity_vocab,
        "resource_vocab": resource_vocab,
        "reverse_activity_vocab": reverse_activity_vocab,
        "reverse_resource_vocab": reverse_resource_vocab,
        "feature_configs": feature_configs,
        "feature_layout": feature_layout,
        "feature_encoder": feature_encoder,
        "activity_feature": activity_feature,
        "extra_vocabs": feature_encoder.categorical_vocabs,
        "ordered_extra_keys": [cfg.name for cfg in feature_configs],
        "normalization_stats": normalization_stats,
        "input_dim": feature_layout.num_dim,
        "graph_builder": graph_builder,
        "prefix_policy": prefix_policy,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


def _compute_class_weights(train_dataset: Sequence[Data], num_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights with 0.0 for absent classes."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for sample in train_dataset:
        y_idx = int(sample.y.view(-1)[0].item())
        if 0 <= y_idx < num_classes:
            counts[y_idx] += 1.0

    total_samples = float(np.sum(counts))
    weights = np.zeros(num_classes, dtype=np.float32)
    if total_samples > 0.0:
        for idx in range(num_classes):
            if counts[idx] > 0.0:
                weights[idx] = float(total_samples / (num_classes * counts[idx]))
            else:
                weights[idx] = 0.0

    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    """Parse CLI args, wire dependencies, and run model training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run MVP1 next-activity training pipeline.")
    parser.add_argument("--config", default="configs/permit_log.yaml", help="YAML experiment config path or filename.")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    config = load_yaml_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    experiment_cfg = config.get("experiment", {})
    tracking_cfg = config.get("tracking", {})

    prepared = prepare_data(config)
    activity_vocab = prepared["activity_vocab"]
    resource_vocab = prepared["resource_vocab"]

    logger.info("Built vocabularies: activity_vocab=%d, resource_vocab=%d", len(activity_vocab), len(resource_vocab))

    # input_dim = базові one-hot + часові фічі + typed extra-фічі (узгоджено з graph_builder).
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    output_dim = len(activity_vocab)
    dropout = float(model_cfg.get("dropout", 0.2))
    pooling_strategy = str(model_cfg.get("pooling_strategy", "global_mean")).strip().lower()
    feature_layout = prepared["feature_layout"]

    model_registry: Dict[str, Type[BaseGNN]] = {
        "BaselineGCN": BaselineGCN,
        "BaselineGATv2": BaselineGATv2,
    }
    model_type = str(model_cfg.get("type", "BaselineGCN"))
    model_cls = model_registry.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported model.type '{model_type}'. Available: {list(model_registry)}")

    model = model_cls(
        feature_layout=feature_layout,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        pooling_strategy=pooling_strategy,
    )

    device = torch.device(str(training_cfg.get("device", "cpu")))
    class_weights = _compute_class_weights(prepared["train_dataset"], output_dim, device)

    run_name = str(experiment_cfg.get("name", f"{model_type}_seed{seed}"))
    tracker = None
    if bool(tracking_cfg.get("enabled", False)):
        tracker = MLflowTracker(
            experiment_name=str(experiment_cfg.get("project", "DefaultExperiment")),
            run_name=run_name,
            tracking_uri=tracking_cfg.get("uri"),
        )

    trainer_config: Dict[str, Any] = {
        **training_cfg,
        "mapping_config": prepared["mapping_config"],
        "data_config": prepared["data_config"],
        "model_config": model_cfg,
        "experiment_config": experiment_cfg,
        "tracking_config": tracking_cfg,
        "config_path": str(config_path),
        "feature_configs": [
            {
                "name": item.name,
                "encoding": list(item.encoding),
                "source": item.source,
                "dtype": item.dtype,
                "role": item.role,
            }
            for item in prepared["feature_configs"]
        ],
        "feature_layout": {
            "num_cat_features": len(prepared["feature_layout"].cat_feature_names),
            "num_num_channels": int(prepared["feature_layout"].num_dim),
            "cat_feature_names": list(prepared["feature_layout"].cat_feature_names),
        },
        "seed": seed,
        "class_weight_cap": float(training_cfg.get("class_weight_cap", 50.0)),
        "retrain": bool(training_cfg.get("retrain", False)),
        "checkpoint_dir": str(training_cfg.get("checkpoint_dir", "checkpoints")),
    }

    trainer = ModelTrainer(
        xes_adapter=XESAdapter(),
        prefix_policy=PrefixPolicy(),
        graph_builder=prepared["graph_builder"],
        model=model,
        log_path=prepared["log_path"],
        config=trainer_config,
        tracker=tracker,
        class_weights=class_weights,
    )

    try:
        results = trainer.run()
        test_metrics = results.get("test_metrics", {})

        print("=== Final Test Metrics ===")
        for key, value in test_metrics.items():
            print(f"{key}: {value}")
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
