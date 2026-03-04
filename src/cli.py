"""Composition Root for MVP1 training pipeline."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (Clean Architecture, MVP1 scope) і розділ 5 (потік RawTrace -> PrefixSlice -> GraphTensorContract)
# - ARCHITECTURE_RULES.md -> розділ 2-4 (Application orchestration через порти)
# - DATA_FLOWS_MVP1.MD -> розділ 3.1 (Training Pipeline)
# - EVF_MVP1.MD -> розділ 3 (Strict Temporal Split) і розділ 4 (фінальні test метрики)

from __future__ import annotations

import argparse
from typing import Any, Dict, Tuple

import yaml

from src.adapters.ingestion.xes_adapter import XESAdapter
from src.application.use_cases.trainer import ModelTrainer
from src.domain.models.baseline_gcn import BaselineGCN
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.prefix_policy import PrefixPolicy


def _load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load runtime configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a YAML mapping at top level.")
    return loaded


def _build_vocabularies(log_path: str, config: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build activity/resource vocabularies from one full pass over normalized traces."""
    adapter = XESAdapter()
    mapping_config = config.get("mapping_config", {})

    unique_activities = set()
    unique_resources = set()

    # Один прохід по XES через стрімінговий адаптер для побудови словників.
    for trace in adapter.read(log_path, mapping_config):
        for event in trace.events:
            unique_activities.add(event.activity_id)
            unique_resources.add(event.resource_id)

    # Індекс 0 зарезервовано для <UNK>, тому реальні значення починаються з 1.
    activity_vocab = {activity: idx for idx, activity in enumerate(sorted(unique_activities), start=1)}
    resource_vocab = {resource: idx for idx, resource in enumerate(sorted(unique_resources), start=1)}

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
    parser = argparse.ArgumentParser(description="Run MVP1 next-activity training pipeline.")
    parser.add_argument("--log_path", required=True, help="Path to input XES log file.")
    parser.add_argument("--config_path", required=True, help="Path to YAML config (model_params.yaml).")
    args = parser.parse_args()

    config = _load_yaml_config(args.config_path)

    # Composition Root: ініціалізація залежностей і зв'язування шарів.
    xes_adapter = XESAdapter()
    prefix_policy = PrefixPolicy()

    activity_vocab, resource_vocab = _build_vocabularies(args.log_path, config)
    normalization_stats = _build_normalization_stats()

    graph_builder = BaselineGraphBuilder(
        activity_vocab=activity_vocab,
        resource_vocab=resource_vocab,
        normalization_stats=normalization_stats,
    )

    # input_dim = |V_act| + 1(UNK) + |V_res| + 1(UNK) + 3.
    input_dim = len(activity_vocab) + 1 + len(resource_vocab) + 1 + 3
    hidden_dim = int(config.get("hidden_dim", 64))
    output_dim = len(activity_vocab) + 1

    model = BaselineGCN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=float(config.get("dropout", 0.2)),
    )

    trainer = ModelTrainer(
        xes_adapter=xes_adapter,
        prefix_policy=prefix_policy,
        graph_builder=graph_builder,
        model=model,
        log_path=args.log_path,
        config=config,
        tracker=None,
    )

    results = trainer.run()
    test_metrics = results.get("test_metrics", {})

    print("=== Final Test Metrics ===")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
