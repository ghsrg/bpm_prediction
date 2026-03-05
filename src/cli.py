"""Composition Root for MVP1 training pipeline."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (Clean Architecture, MVP1 scope) і розділ 5 (послідовність потоків)
# - ARCHITECTURE_RULES.md -> розділ 2-4 (Application orchestration через порти)
# - DATA_FLOWS_MVP1.MD -> розділ 3.1 (Training Pipeline)
# - EVF_MVP1.MD -> розділ 3 (Strict Temporal Split) і розділ 5 (MLflow tracking)

from __future__ import annotations

import argparse
from pathlib import Path
import logging
import random
from typing import Any, Dict, List, Sequence, Tuple, Type

import numpy as np
import torch
import yaml
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
    """Load runtime configuration from YAML file."""
    config_path = _resolve_config_path(config_arg)
    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a YAML mapping at top level.")
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


def _build_extra_feature_artifacts(
    traces: Sequence[RawTrace],
    extra_trace_keys: Sequence[str],
    extra_event_keys: Sequence[str],
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Infer extra feature schema and vocabularies for categorical attributes."""
    ordered_extra_keys = list(dict.fromkeys([*extra_trace_keys, *extra_event_keys]))

    # Детерміновано фіксуємо тип ознаки за першим не-None значенням.
    feature_types: Dict[str, str] = {}
    # Для string-категорій збираємо унікальні значення для one-hot словника.
    categorical_values: Dict[str, set[str]] = {key: set() for key in ordered_extra_keys}

    for trace in traces:
        for key in extra_trace_keys:
            value = trace.trace_attributes.get(key)
            if value is not None and key not in feature_types:
                if isinstance(value, bool):
                    feature_types[key] = "bool"
                elif isinstance(value, (int, float)):
                    feature_types[key] = "numeric"
                else:
                    feature_types[key] = "categorical"
            if isinstance(value, str) and value:
                categorical_values[key].add(value)

        for event in trace.events:
            for key in extra_event_keys:
                value = event.extra.get(key)
                if value is not None and key not in feature_types:
                    if isinstance(value, bool):
                        feature_types[key] = "bool"
                    elif isinstance(value, (int, float)):
                        feature_types[key] = "numeric"
                    else:
                        feature_types[key] = "categorical"
                if isinstance(value, str) and value:
                    categorical_values[key].add(value)

    extra_vocabs: Dict[str, Dict[str, int]] = {}
    for key in ordered_extra_keys:
        if feature_types.get(key) == "categorical":
            extra_vocabs[key] = {val: idx for idx, val in enumerate(sorted(categorical_values[key]))}

    return ordered_extra_keys, extra_vocabs


def _build_normalization_stats() -> Dict[str, Dict[str, float]]:
    """Create MVP1 placeholder normalization stats (mu=0, sigma=1)."""
    return {
        "duration": {"mu": 0.0, "sigma": 1.0},
        "time_since_case_start": {"mu": 0.0, "sigma": 1.0},
        "time_since_previous_event": {"mu": 0.0, "sigma": 1.0},
    }


def _strict_temporal_split(traces: Sequence[RawTrace]) -> Tuple[List[RawTrace], List[RawTrace], List[RawTrace]]:
    """Apply strict chronological split (70/10/20) by first event timestamp."""
    traces_with_events = [trace for trace in traces if trace.events]
    ordered = sorted(traces_with_events, key=lambda tr: tr.events[0].timestamp)

    total = len(ordered)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.1)
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

    xes_adapter = XESAdapter()
    prefix_policy = PrefixPolicy()

    traces = list(xes_adapter.read(log_path, mapping_cfg))
    feature_configs = parse_feature_configs(config)
    feature_encoder = FeatureEncoder(feature_configs=feature_configs, traces=traces)
    feature_layout = feature_encoder.feature_layout

    activity_vocab, resource_vocab, activity_feature = _extract_base_vocabularies(feature_encoder)
    reverse_activity_vocab = {idx: key for key, idx in activity_vocab.items()}
    reverse_resource_vocab = {idx: key for key, idx in resource_vocab.items()}

    traces = list(xes_adapter.read(log_path, mapping_cfg))

    xes_cfg = mapping_cfg.get("xes_adapter", mapping_cfg)
    extra_trace_keys = [str(k) for k in xes_cfg.get("extra_trace_keys", [])]
    extra_event_keys = [str(k) for k in xes_cfg.get("extra_event_keys", [])]
    ordered_extra_keys, extra_vocabs = _build_extra_feature_artifacts(
        traces=traces,
        extra_trace_keys=extra_trace_keys,
        extra_event_keys=extra_event_keys,
    )

    normalization_stats = _build_normalization_stats()
    graph_builder = BaselineGraphBuilder(feature_encoder=feature_encoder)
    train_traces, val_traces, test_traces = _strict_temporal_split(traces)

    train_dataset = _build_graph_dataset(train_traces, prefix_policy, graph_builder)
    val_dataset = _build_graph_dataset(val_traces, prefix_policy, graph_builder)
    test_dataset = _build_graph_dataset(test_traces, prefix_policy, graph_builder)

    return {
        "log_path": log_path,
        "mapping_config": mapping_cfg,
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
    )

    device = torch.device(str(training_cfg.get("device", "cpu")))
    class_weights = _compute_class_weights(prepared["train_dataset"], output_dim, device)

    run_name = f"{model_type}_seed{seed}"
    tracker = None
    if bool(tracking_cfg.get("enabled", False)):
        tracker = MLflowTracker(
            experiment_name=str(experiment_cfg.get("name", "DefaultExperiment")),
            run_name=run_name,
            tracking_uri=tracking_cfg.get("uri"),
        )

    trainer_config: Dict[str, Any] = {
        **training_cfg,
        "mapping_config": prepared["mapping_config"],
        "seed": seed,
        "class_weight_cap": float(training_cfg.get("class_weight_cap", 50.0)),
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
