from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


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


def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def test_dynamic_graph_builder_keeps_unseen_topology_candidate_and_edges(mock_feature_configs):
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", ["A", "B"])],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v2",
        ProcessStructureDTO(
            version="v2",
            allowed_edges=[("A", "C"), ("C", "B")],
            nodes=[
                {"id": "A", "bpmn_tag": "task", "type": "task", "label": "A"},
                {"id": "C", "bpmn_tag": "task", "type": "task", "label": "C"},
                {"id": "B", "bpmn_tag": "task", "type": "task", "label": "B"},
            ],
        ),
    )
    prefix = PrefixSlice(
        case_id="eval",
        process_version="v2",
        prefix_events=[_event(0, "A")],
        target_event=_event(1, "C"),
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        candidate_identity_mode="topology_native",
    ).build_graph(prefix)

    assert contract["target_label"] == "C"
    assert tuple(contract["candidate_labels"]) == ("A", "B", "C")
    assert tuple(contract["candidate_ids"]) == ("A", "B", "C")
    assert contract["candidate_class_index"].tolist() == [
        encoder.categorical_vocabs["concept:name"]["A"],
        encoder.categorical_vocabs["concept:name"]["B"],
        -1,
    ]
    assert contract["candidate_is_unseen"].tolist() == [False, False, True]
    assert contract["struct_node_to_candidate_index"].tolist() == [0, 1, 2]
    assert int(contract["structural_edge_index"].shape[1]) == 2
    assert contract["candidate_allowed_target_mask"].tolist() == [False, False, True]


def test_dynamic_graph_builder_fixed_vocab_bridge_keeps_candidate_axis_stable_for_unseen_topology(
    mock_feature_configs,
):
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", ["A", "B"])],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v2",
        ProcessStructureDTO(
            version="v2",
            allowed_edges=[("A", "C"), ("C", "B")],
            nodes=[
                {"id": "A", "bpmn_tag": "task", "type": "task", "label": "A"},
                {"id": "C", "bpmn_tag": "task", "type": "task", "label": "C"},
                {"id": "B", "bpmn_tag": "task", "type": "task", "label": "B"},
            ],
        ),
    )
    prefix = PrefixSlice(
        case_id="eval",
        process_version="v2",
        prefix_events=[_event(0, "A")],
        target_event=_event(1, "C"),
    )

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        candidate_identity_mode="fixed_vocab_bridge",
    ).build_graph(prefix)

    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert contract["candidate_allowed_target_mask"].numel() == len(activity_vocab)
    assert contract["candidate_class_index"].tolist() == list(range(len(activity_vocab)))
    assert tuple(contract["candidate_labels"]) == tuple(
        label for label, _idx in sorted(activity_vocab.items(), key=lambda item: item[1])
    )
    assert "C" not in contract["candidate_labels"]


def test_fixed_vocab_bridge_candidate_masks_collate_across_topology_versions(mock_feature_configs):
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", ["A", "B"])],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v1",
        ProcessStructureDTO(
            version="v1",
            allowed_edges=[("A", "B")],
            nodes=[
                {"id": "A", "bpmn_tag": "task", "type": "task", "label": "A"},
                {"id": "B", "bpmn_tag": "task", "type": "task", "label": "B"},
            ],
        ),
    )
    repository.save_process_structure(
        "v2",
        ProcessStructureDTO(
            version="v2",
            allowed_edges=[("A", "C"), ("C", "B")],
            nodes=[
                {"id": "A", "bpmn_tag": "task", "type": "task", "label": "A"},
                {"id": "C", "bpmn_tag": "task", "type": "task", "label": "C"},
                {"id": "B", "bpmn_tag": "task", "type": "task", "label": "B"},
            ],
        ),
    )
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        candidate_identity_mode="fixed_vocab_bridge",
    )

    contracts = [
        builder.build_graph(
            PrefixSlice(
                case_id="eval_v1",
                process_version="v1",
                prefix_events=[_event(0, "A")],
                target_event=_event(1, "B"),
            )
        ),
        builder.build_graph(
            PrefixSlice(
                case_id="eval_v2",
                process_version="v2",
                prefix_events=[_event(0, "A")],
                target_event=_event(1, "C"),
            )
        ),
    ]
    data = [
        Data(
            x_cat=contract["x_cat"],
            x_num=contract["x_num"],
            edge_index=contract["edge_index"],
            edge_type=contract["edge_type"],
            y=contract["y"],
            num_nodes=int(contract["num_nodes"]),
            candidate_allowed_target_mask=contract["candidate_allowed_target_mask"].unsqueeze(0),
        )
        for contract in contracts
    ]

    batch = next(iter(DataLoader(data, batch_size=2)))

    assert batch.candidate_allowed_target_mask.shape == (2, len(encoder.categorical_vocabs[encoder.activity_feature_name]))
