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
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
import yaml

from src.adapters.ingestion.xes_adapter import XESAdapter
from src.application.use_cases.trainer import ModelTrainer
from src.domain.models.base_gnn import BaseGNN
from src.domain.models.baseline_gat import BaselineGATv2
from src.domain.models.baseline_gcn import BaselineGCN
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
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


def _load_yaml_config(config_arg: str) -> Dict[str, Any]:
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


def _build_vocabularies(log_path: str, mapping_config: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build activity/resource vocabularies from one full pass over normalized traces."""
    adapter = XESAdapter()

    # Індекс 0 зарезервовано для <UNK> для обох словників.
    activity_vocab: Dict[str, int] = {"<UNK>": 0}
    resource_vocab: Dict[str, int] = {"<UNK>": 0}

    # Один прохід по XES через стрімінговий адаптер для побудови словників.
    for trace in adapter.read(log_path, mapping_config):
        for event in trace.events:
            activity = event.activity_id
            resource = event.resource_id

            if activity not in activity_vocab:
                activity_vocab[activity] = len(activity_vocab)
            if resource not in resource_vocab:
                resource_vocab[resource] = len(resource_vocab)

    return activity_vocab, resource_vocab


def _build_normalization_stats() -> Dict[str, Dict[str, float]]:
    """Create MVP1 placeholder normalization stats (mu=0, sigma=1)."""
    return {
        "duration": {"mu": 0.0, "sigma": 1.0},
        "time_since_case_start": {"mu": 0.0, "sigma": 1.0},
        "time_since_previous_event": {"mu": 0.0, "sigma": 1.0},
    }


def main() -> None:
    """Parse CLI args, wire dependencies, and run model training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run MVP1 next-activity training pipeline.")
    parser.add_argument("--config", default="configs/permit_log.yaml", help="YAML experiment config path or filename.")
    args = parser.parse_args()

    config = _load_yaml_config(args.config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    mapping_cfg = config.get("mapping", {})
    experiment_cfg = config.get("experiment", {})
    tracking_cfg = config.get("tracking", {})

    log_path = str(data_cfg.get("log_path", ""))
    if not log_path:
        raise ValueError("Config must define data.log_path")

    # Composition Root: ініціалізація залежностей і зв'язування шарів.
    xes_adapter = XESAdapter()
    prefix_policy = PrefixPolicy()

    activity_vocab, resource_vocab = _build_vocabularies(log_path, mapping_cfg)
    logger.info("Built vocabularies: activity_vocab=%d, resource_vocab=%d", len(activity_vocab), len(resource_vocab))
    normalization_stats = _build_normalization_stats()

    graph_builder = BaselineGraphBuilder(
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        normalization_stats=normalization_stats,
    )

    # input_dim = |V_act| + |V_res| + 3 (UNK вже включений у vocab як індекс 0).
    input_dim = len(activity_vocab) + len(resource_vocab) + 3
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    output_dim = len(activity_vocab)
    dropout = float(model_cfg.get("dropout", 0.2))

    model_registry: Dict[str, Type[BaseGNN]] = {
        "BaselineGCN": BaselineGCN,
        "BaselineGATv2": BaselineGATv2,
    }
    model_type = str(model_cfg.get("type", "BaselineGCN"))
    model_cls = model_registry.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported model.type '{model_type}'. Available: {list(model_registry)}")

    model = model_cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
    )

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
        "mapping_config": mapping_cfg,
        "seed": seed,
    }

    trainer = ModelTrainer(
        xes_adapter=xes_adapter,
        prefix_policy=prefix_policy,
        graph_builder=graph_builder,
        model=model,
        log_path=log_path,
        config=trainer_config,
        tracker=tracker,
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
