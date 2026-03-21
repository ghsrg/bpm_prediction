from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

import pytest
import torch
from torch import nn

from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        return iter([])


class _TrainableBinaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor([1.5, -1.5], dtype=torch.float32))

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return self.logit_bias.unsqueeze(0).repeat(num_graphs, 1)


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
        extra={"concept:name": activity, "org:resource": "R1", "amount": float(idx + 1)},
        activity_instance_id=f"ai_{idx}_{activity}",
    )


def _trace(case_id: str, version: str, activities: list[str], base_ts: float) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act, base_ts + float(i * 60)) for i, act in enumerate(activities)],
        trace_attributes={},
    )


def _repo_with_stats() -> InMemoryNetworkXRepository:
    repo = InMemoryNetworkXRepository()
    dto = ProcessStructureDTO(
        version="v1",
        allowed_edges=[("Start", "Approve"), ("Approve", "End")],
        nodes=[
            {"id": "Start", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
            {"id": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "End", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
        ],
        metadata={
            "knowledge_version": "k000321",
            "as_of_ts": "2026-03-21T10:30:00+00:00",
            "stats_index": {
                "node": {
                    "all_time.version.exec_count": {
                        "Start": 10.0,
                        "Approve": 20.0,
                        "End": 30.0,
                    }
                },
                "edge": {
                    "all_time.version.transition_probability": {
                        "Start|||Approve": 0.25,
                        "Approve|||End": 0.75,
                    }
                },
                "global": {},
            },
        },
    )
    repo.save_process_structure("v1", dto, process_name="dataset_a")
    return repo


def test_stats_values_are_mapped_to_expected_nodes_edges_and_forward_contract(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"], base_ts=1700000000.0)]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    repo = _repo_with_stats()
    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repo,
        process_name="dataset_a",
        graph_feature_mapping={
            "enabled": True,
            "node_numeric": [
                {
                    "name": "node_exec_count_v",
                    "metric": "exec_count",
                    "window": "all_time",
                    "scope": "version",
                    "default": -1.0,
                    "encoding": ["identity"],
                }
            ],
            "edge_weight": {
                "metric": "transition_probability",
                "window": "all_time",
                "scope": "version",
                "default": 1.0,
                "encoding": ["identity"],
            },
        },
    )

    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=PrefixPolicy(),
        graph_builder=builder,
        model=_TrainableBinaryModel(),
        log_path="unused.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "batch_size": 1,
            "experiment_config": {"drift_window_size": 2},
        },
    )

    loader = trainer._build_loader(traces, shuffle=False)
    data = next(iter(loader))
    contract = trainer._data_to_contract(data)

    assert "struct_x" in contract
    struct_x = contract["struct_x"]
    assert struct_x is not None
    activity_vocab = encoder.categorical_vocabs[encoder.activity_feature_name]
    assert float(struct_x[int(activity_vocab["Start"]), 0]) == pytest.approx(10.0, abs=1e-6)
    assert float(struct_x[int(activity_vocab["Approve"]), 0]) == pytest.approx(20.0, abs=1e-6)
    assert float(struct_x[int(activity_vocab["End"]), 0]) == pytest.approx(30.0, abs=1e-6)

    reverse_vocab = {int(idx): token for token, idx in activity_vocab.items()}
    edge_index = contract["structural_edge_index"]
    edge_weight = contract["structural_edge_weight"]
    observed_weights: dict[tuple[str, str], float] = {}
    for col in range(int(edge_index.size(1))):
        src = reverse_vocab[int(edge_index[0, col].item())]
        dst = reverse_vocab[int(edge_index[1, col].item())]
        observed_weights[(src, dst)] = float(edge_weight[col].item())

    assert observed_weights[("Start", "Approve")] == pytest.approx(0.25, abs=1e-6)
    assert observed_weights[("Approve", "End")] == pytest.approx(0.75, abs=1e-6)

    assert contract.get("stats_snapshot_versions") is not None
    assert "k000321" in contract["stats_snapshot_versions"]
    assert contract.get("stats_snapshot_as_of_ts_batch") is not None
    assert "2026-03-21T10:30:00+00:00" in contract["stats_snapshot_as_of_ts_batch"]


class _AsOfProbePort:
    def __init__(self, dto: ProcessStructureDTO) -> None:
        self.dto = dto
        self.called_with_version: str | None = None
        self.called_with_process_name: str | None = None
        self.called_with_as_of: datetime | None = None

    def get_process_structure_as_of(self, version: str, process_name: str | None = None, as_of_ts: datetime | None = None):
        self.called_with_version = version
        self.called_with_process_name = process_name
        self.called_with_as_of = as_of_ts
        return self.dto

    def get_process_structure(self, version: str, process_name: str | None = None):
        _ = version
        _ = process_name
        return self.dto


def test_strict_asof_uses_prefix_last_event_timestamp(mock_feature_configs):
    traces = [_trace("c1", "v1", ["Start", "Approve", "End"], base_ts=1700100000.0)]
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    dto = ProcessStructureDTO(version="v1", allowed_edges=[("Approve", "End")], metadata={})
    probe_repo = _AsOfProbePort(dto=dto)

    builder = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=probe_repo,
        process_name="dataset_probe",
        stats_time_policy="strict_asof",
    )
    prefix = PrefixSlice(
        case_id="probe_case",
        process_version="v1",
        prefix_events=[
            _event(0, "Start", 1700100000.0),
            _event(1, "Approve", 1700101234.0),
        ],
        target_event=_event(2, "End", 1700101300.0),
    )

    contract = builder.build_graph(prefix)
    assert contract["allowed_target_mask"] is not None
    assert probe_repo.called_with_version == "v1"
    assert probe_repo.called_with_process_name == "dataset_probe"
    assert probe_repo.called_with_as_of is not None
    assert float(probe_repo.called_with_as_of.timestamp()) == pytest.approx(1700101234.0, abs=1e-6)
