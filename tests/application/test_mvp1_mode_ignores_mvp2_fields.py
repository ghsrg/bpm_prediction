from __future__ import annotations

from typing import Iterator

import pytest

from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.event_record import EventRecord
from src.domain.entities.feature_config import FeatureLayout
from src.domain.entities.raw_trace import RawTrace
from src.domain.models.baseline_gcn import BaselineGCN


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        return iter([])


class _DummyPrefixPolicy:
    def generate_slices(self, trace: RawTrace):
        _ = trace
        return []


class _DummyGraphBuilder:
    def build_graph(self, prefix):
        _ = prefix
        raise RuntimeError("Not used in this test.")


def _build_trace(ts: float, case_id: str) -> RawTrace:
    event = EventRecord(
        activity_id="A",
        timestamp=ts,
        resource_id="R",
        lifecycle="complete",
        position_in_trace=0,
        duration=1.0,
        time_since_case_start=0.0,
        time_since_previous_event=0.0,
        extra={"concept:name": "A"},
        activity_instance_id=f"ai_{case_id}",
    )
    return RawTrace(case_id=case_id, process_version="v1", events=[event], trace_attributes={})


@pytest.mark.mvp1_regression
@pytest.mark.parametrize("mode", ["train", "eval_drift", "eval_cross_dataset"])
def test_mvp1_modes_ignore_mvp2_fields_in_experiment_config(mode: str):
    model = BaselineGCN(
        feature_layout=FeatureLayout(cat_features={}, cat_feature_names=[], num_dim=1),
        hidden_dim=8,
        output_dim=3,
        dropout=0.0,
    )
    trainer = ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=_DummyGraphBuilder(),
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": mode,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "experiment_config": {
                "split_strategy": "temporal",
                "train_ratio": 0.7,
                "fraction": 1.0,
                "split_ratio": [0.7, 0.2, 0.1],
                # MVP2-specific knobs should not affect MVP1 prep flow.
                "structure_mode": "epokg",
                "kappa_conditioning": "static",
                "use_structural_mask": True,
            },
        },
    )

    traces = [_build_trace(float(i), f"c{i}") for i in range(10)]
    prepared = trainer._prepare_data(traces, mode=mode)
    split = trainer._prepare_split_data(prepared)

    assert isinstance(prepared, list)
    assert len(split.train) + len(split.val) + len(split.test) == len(prepared)

