import os
import sys
import psutil
import torch
import gc
from datetime import datetime, timezone

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cli import load_yaml_config, _apply_experiment_switch_overrides
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.entities.feature_config import parse_feature_configs
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.event_record import EventRecord

def print_memory(label):
    process = psutil.Process()
    rss = process.memory_info().rss / (1024 * 1024)
    print(f"[{label}] RSS: {rss:.2f} MB")

def _event(idx: int, activity: str, ts: float) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=ts,
        resource_id="R1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity, "org:resource": "R1"},
        activity_instance_id=f"ai_{idx}_{activity}",
    )

def main():
    print_memory("Start")
    config_path = "configs/experiments/ui_run_feacvhua.yaml"
    if not os.path.exists(config_path):
        print("Config not found")
        return
    config = load_yaml_config(config_path)
    config = _apply_experiment_switch_overrides(config)
    repository = InMemoryNetworkXRepository()
    
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("A", "B"), ("B", "C")],
        nodes=[
            {"id": "A", "bpmn_tag": "task", "type": "task", "label": "A"},
            {"id": "B", "bpmn_tag": "task", "type": "task", "label": "B"},
            {"id": "C", "bpmn_tag": "task", "type": "task", "label": "C"},
        ],
        metadata={
            "stats_contract": {
                "identity": {
                    "knowledge_version": "k000001",
                    "as_of_ts": "2026-05-01T00:00:00Z"
                }
            }
        }
    )
    repository.save_process_structure_snapshot("v1", dto, as_of_ts=datetime(2026, 5, 1, tzinfo=timezone.utc))
    
    feature_configs = parse_feature_configs(config)
    encoder = FeatureEncoder(
        feature_configs=feature_configs,
        traces=[],
    )
    # Manually populate vocabs to avoid fit overhead
    encoder.categorical_vocab_sizes = {
        "concept:name": 4,
        "org:resource": 2
    }
    encoder.categorical_vocabs = {
        "concept:name": {"<UNK>": 0, "A": 1, "B": 2, "C": 3},
        "org:resource": {"UNKNOWN": 0, "R1": 1}
    }
    
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        process_name="test_process",
        graph_feature_mapping=config.get("mapping", {}).get("graph_feature_mapping", {}),
        stats_time_policy="strict_asof",
        on_missing_asof_snapshot="use_base",
        cache_policy="full",
        candidate_identity_mode="topology_native",
    )
    
    # Warm up
    prefix = PrefixSlice(
        case_id="case1",
        process_version="v1",
        prefix_events=[_event(0, "A", 1700100000.0), _event(1, "B", 1700100001.0)],
        target_event=_event(2, "C", 1700100002.0)
    )
    builder.build_graph(prefix)
    gc.collect()
    print_memory("After warm-up")
    
    # Loop 10000 times with changing timestamps
    base_ts = 1700100000.0
    for i in range(10000):
        prefix = PrefixSlice(
            case_id=f"case_{i}",
            process_version="v1",
            prefix_events=[_event(0, "A", base_ts + i), _event(1, "B", base_ts + i + 1)],
            target_event=_event(2, "C", base_ts + i + 2)
        )
        builder.build_graph(prefix)
        if i % 1000 == 0:
            gc.collect()
            print_memory(f"Iteration {i}")
            
    gc.collect()
    print_memory("End")
    print(f"DTO Cache size: {len(builder._dto_cache)}")
    print(f"Topology Cache size: {len(builder._topology_cache)}")

if __name__ == "__main__":
    main()
