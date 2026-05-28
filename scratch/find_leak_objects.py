import os
import sys
import gc
from datetime import datetime, timezone

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cli import load_yaml_config, _apply_experiment_switch_overrides, _build_trace_adapter
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.entities.feature_config import parse_feature_configs
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.event_record import EventRecord

def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700100000 + idx),
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
    config = load_yaml_config("configs/experiments/ui_run_feacvhua.yaml")
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
    
    # Create real FeatureEncoder with dummy trace
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
    
    prefix = PrefixSlice(
        case_id="case1",
        process_version="v1",
        prefix_events=[_event(0, "A"), _event(1, "B")],
        target_event=_event(2, "C")
    )
    
    # Warm up
    builder.build_graph(prefix)
    gc.collect()
    
    # Snapshot 1
    gc.collect()
    objs_before = gc.get_objects()
    types_before = {}
    for obj in objs_before:
        t = type(obj).__name__
        types_before[t] = types_before.get(t, 0) + 1
        
    # Loop 1000 times
    for _ in range(1000):
        builder.build_graph(prefix)
        
    # Snapshot 2
    gc.collect()
    objs_after = gc.get_objects()
    types_after = {}
    for obj in objs_after:
        t = type(obj).__name__
        types_after[t] = types_after.get(t, 0) + 1
        
    print("=== Object type count differences ===")
    diffs = []
    for t in set(types_before.keys()) | set(types_after.keys()):
        count_before = types_before.get(t, 0)
        count_after = types_after.get(t, 0)
        diff = count_after - count_before
        if diff > 0:  # print any growth
            diffs.append((t, count_before, count_after, diff))
            
    diffs.sort(key=lambda x: x[3], reverse=True)
    for t, cb, ca, d in diffs:
        print(f"{t}: {cb} -> {ca} (diff: +{d})")

    # Let's inspect some of the leaking objects of the top type
    top_type = diffs[0][0] if diffs else None
    if top_type:
        print(f"\nLeaking objects of type {top_type}:")
        count = 0
        for obj in objs_after:
            if type(obj).__name__ == top_type and obj not in objs_before:
                print(repr(obj)[:200])
                count += 1
                if count >= 10:
                    break

if __name__ == "__main__":
    main()
