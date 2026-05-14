from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.raw_trace import RawTrace


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict):
        _ = file_path
        _ = mapping_config
        return []


class _DummyPrefixPolicy:
    def __init__(self, prefix: PrefixSlice) -> None:
        self.prefix = prefix

    def generate_slices(self, trace: RawTrace):
        _ = trace
        return [self.prefix]


class _DummyGraphBuilder:
    def __init__(self, contract: dict) -> None:
        self.contract = contract

    def build_graph(self, prefix: PrefixSlice):
        _ = prefix
        return self.contract


class _PassthroughModel(nn.Module):
    def forward(self, contract):  # pragma: no cover - not used in these tests
        return torch.zeros((1, 2), dtype=torch.float32)


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


def _prefix() -> PrefixSlice:
    return PrefixSlice(
        case_id="c1",
        process_version="v1",
        prefix_events=[_event(0, "Start")],
        target_event=_event(1, "End"),
    )


def _trace() -> RawTrace:
    return RawTrace(
        case_id="c1",
        process_version="v1",
        events=[_event(0, "Start"), _event(1, "End")],
        trace_attributes={},
    )


def _contract(prefix_state: torch.Tensor) -> dict:
    return {
        "x_cat": torch.zeros((1, 0), dtype=torch.long),
        "x_num": torch.ones((1, 1), dtype=torch.float32),
        "edge_index": torch.zeros((2, 0), dtype=torch.long),
        "edge_type": torch.zeros((0,), dtype=torch.long),
        "y": torch.tensor([1], dtype=torch.long),
        "batch": torch.zeros((1,), dtype=torch.long),
        "num_nodes": 1,
        "struct_x": torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
        "structural_edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "structural_edge_weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "struct_node_to_class_index": torch.tensor([0, 1, 2], dtype=torch.long),
        "struct_prefix_state_x": prefix_state,
    }


def _trainer(contract: dict | None = None) -> ModelTrainer:
    prefix = _prefix()
    return ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(prefix),
        graph_builder=_DummyGraphBuilder(contract or _contract(torch.zeros((3, 6), dtype=torch.float32))),
        model=_PassthroughModel(),
        log_path="in_memory.xes",
        config={"batch_size": 2, "device": "cpu", "show_progress": False, "tqdm_disable": True},
    )


def test_build_loader_copies_struct_prefix_state_x():
    prefix_state = torch.arange(18, dtype=torch.float32).view(3, 6)
    trainer = _trainer(_contract(prefix_state))

    loader = trainer._build_loader([_trace()], shuffle=False)
    data = next(iter(loader))

    assert hasattr(data, "struct_prefix_state_x")
    assert data.struct_prefix_state_x.shape == torch.Size([3, 6])
    assert torch.equal(data.struct_prefix_state_x, prefix_state)


def test_data_to_contract_stacks_batched_struct_prefix_state_x():
    sample_a = Data(**_contract(torch.ones((3, 6), dtype=torch.float32)))
    sample_b = Data(**_contract(torch.full((3, 6), 2.0, dtype=torch.float32)))
    batch = next(iter(DataLoader([sample_a, sample_b], batch_size=2, shuffle=False)))
    trainer = _trainer()

    contract = trainer._data_to_contract(batch)

    assert "struct_prefix_state_x" in contract
    assert contract["struct_prefix_state_x"].shape == torch.Size([2, 3, 6])
    assert torch.equal(contract["struct_prefix_state_x"][0], sample_a.struct_prefix_state_x)
    assert torch.equal(contract["struct_prefix_state_x"][1], sample_b.struct_prefix_state_x)
