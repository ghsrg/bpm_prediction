from __future__ import annotations

import json
from typing import Iterator

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.application.services.outcome_partition_audit import (
    AuditObservation,
    aggregate_outcomes,
    classify_observation,
)
from src.application.use_cases.trainer import DriftInferenceRecords, ModelTrainer, SplitData
from src.domain.entities.candidate_prediction import CandidatePredictionOutput
from src.domain.entities.event_record import EventRecord
from src.domain.entities.raw_trace import RawTrace
from src.infrastructure.runtime.progress_events import PROGRESS_EVENT_PREFIX


class _FailOnReadAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        raise AssertionError("Adapter read should not be called in one-pass drift tests.")


class _NoopPrefixPolicy:
    def generate_slices(self, trace: RawTrace):
        _ = trace
        return []


class _NoopGraphBuilder:
    def build_graph(self, prefix_slice):
        _ = prefix_slice
        raise AssertionError("Graph builder should not be called in one-pass drift tests.")


class _PredictFromXNumModel(nn.Module):
    def __init__(self, output_dim: int = 3) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, contract):
        pred = contract["x_num"].view(-1).long().clamp(min=0, max=self.output_dim - 1)
        logits = torch.full((int(pred.shape[0]), self.output_dim), -5.0, device=pred.device)
        logits[torch.arange(int(pred.shape[0]), device=pred.device), pred] = 5.0
        return logits


class _CountingLogitModel(_PredictFromXNumModel):
    def __init__(self, output_dim: int = 3) -> None:
        super().__init__(output_dim=output_dim)
        self.forward_calls = 0
        self.raw_logits: list[torch.Tensor] = []

    def forward(self, contract):
        self.forward_calls += 1
        logits = super().forward(contract)
        self.raw_logits.append(logits.detach().cpu().clone())
        return logits


class _UnseenCandidateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 3

    def forward_candidate(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        logits = torch.tensor([[0.0, 5.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        return CandidatePredictionOutput(
            candidate_logits=logits,
            candidate_class_index=torch.tensor([1, -1], dtype=torch.long, device=batch.device),
            node_logits=logits,
            node_to_candidate_index=torch.tensor([0, 1], dtype=torch.long, device=batch.device),
            node_to_class_index=torch.tensor([1, -1], dtype=torch.long, device=batch.device),
            candidate_ids=("node_known", "node_new"),
            candidate_labels=("known_task", "new_task"),
            candidate_is_unseen=torch.tensor([False, True], dtype=torch.bool, device=batch.device),
        )


class _KnownCandidateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 2

    def forward_candidate(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        logits = torch.tensor([[0.0, 5.0]], dtype=torch.float32, device=batch.device).repeat(num_graphs, 1)
        return CandidatePredictionOutput(
            candidate_logits=logits,
            candidate_class_index=torch.tensor([0, 1], dtype=torch.long, device=batch.device),
            node_logits=logits,
            node_to_candidate_index=torch.tensor([0, 1], dtype=torch.long, device=batch.device),
            node_to_class_index=torch.tensor([0, 1], dtype=torch.long, device=batch.device),
            candidate_ids=("node_a", "node_b"),
            candidate_labels=("A", "B"),
            candidate_is_unseen=torch.tensor([False, False], dtype=torch.bool, device=batch.device),
        )


class _TrainableThresholdModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 2
        self.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, contract):
        batch = contract["batch"]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        feature = contract["x_num"].view(num_graphs, -1)[:, 0]
        class_one_logit = feature + self.bias
        return torch.stack((-class_one_logit, class_one_logit), dim=1)


class _FakeTracker:
    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.metrics.append((key, float(value), step))

    def log_param(self, key: str, value) -> None:
        _ = key
        _ = value

    def log_tag(self, key: str, value) -> None:
        _ = key
        _ = value

    def log_artifact(self, local_path: str) -> None:
        _ = local_path

    def log_model(self, model, artifact_path: str) -> None:
        _ = model
        _ = artifact_path


def _event(idx: int) -> EventRecord:
    return EventRecord(
        activity_id=f"A{idx}",
        timestamp=float(1700000000 + idx),
        resource_id="r1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": f"A{idx}", "org:resource": "r1"},
        activity_instance_id=f"ai_{idx}",
    )


def _trace(case_id: str, idx: int) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version="v1",
        events=[_event(idx), _event(idx + 1)],
        trace_attributes={},
    )


def _sample(*, trace_idx: int, target: int, pred: int, mask: list[bool] | None = None) -> Data:
    payload = {
        "x_cat": torch.zeros((1, 1), dtype=torch.long),
        "x_num": torch.tensor([[float(pred)]], dtype=torch.float32),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "edge_type": torch.empty((0,), dtype=torch.long),
        "y": torch.tensor([target], dtype=torch.long),
        "num_nodes": 1,
        "trace_idx": torch.tensor([trace_idx], dtype=torch.long),
        "prefix_idx": torch.tensor([0], dtype=torch.long),
        "prefix_len": torch.tensor([1], dtype=torch.long),
        "process_version_idx": torch.tensor([0], dtype=torch.long),
        "trace_start_ts": torch.tensor([float(1700000000 + trace_idx)], dtype=torch.float64),
        "trace_end_ts": torch.tensor([float(1700000010 + trace_idx)], dtype=torch.float64),
    }
    if mask is not None:
        payload["allowed_target_mask"] = torch.tensor([mask], dtype=torch.bool)
    return Data(**payload)


def _trainer(
    tmp_path,
    *,
    drift_window_size: int = 2,
    drift_window_sliding: int = 1,
    model: nn.Module | None = None,
    finetune_start_ratio: float = 0.0,
) -> ModelTrainer:
    return ModelTrainer(
        xes_adapter=_FailOnReadAdapter(),
        prefix_policy=_NoopPrefixPolicy(),  # type: ignore[arg-type]
        graph_builder=_NoopGraphBuilder(),  # type: ignore[arg-type]
        model=model or _PredictFromXNumModel(output_dim=3),  # type: ignore[arg-type]
        log_path="in_memory.xes",
        config={
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "checkpoint_dir": str(tmp_path),
            "drift_window_size": drift_window_size,
            "drift_window_sliding": drift_window_sliding,
            "experiment_config": {
                "name": "pytest_one_pass_drift",
                "mode": "eval_drift",
                "drift_window_size": drift_window_size,
                "drift_window_sliding": drift_window_sliding,
                "finetune_start_ratio": finetune_start_ratio,
            },
        },
        prepared_data={"idx_to_version": {0: "v1"}},
    )


def _candidate_trainer(tmp_path) -> ModelTrainer:
    return ModelTrainer(
        xes_adapter=_FailOnReadAdapter(),
        prefix_policy=_NoopPrefixPolicy(),  # type: ignore[arg-type]
        graph_builder=_NoopGraphBuilder(),  # type: ignore[arg-type]
        model=_UnseenCandidateModel(),  # type: ignore[arg-type]
        log_path="in_memory.xes",
        config={
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "checkpoint_dir": str(tmp_path),
            "candidate_contract_mode": "candidate_id",
            "candidate_identity_mode": "topology_native",
            "experiment_config": {
                "name": "pytest_candidate_one_pass_drift",
                "mode": "eval_drift",
                "drift_window_size": 1,
                "drift_window_sliding": 1,
            },
        },
        prepared_data={"idx_to_version": {0: "v1"}},
    )


def _known_candidate_trainer(tmp_path) -> ModelTrainer:
    return ModelTrainer(
        xes_adapter=_FailOnReadAdapter(),
        prefix_policy=_NoopPrefixPolicy(),  # type: ignore[arg-type]
        graph_builder=_NoopGraphBuilder(),  # type: ignore[arg-type]
        model=_KnownCandidateModel(),  # type: ignore[arg-type]
        log_path="in_memory.xes",
        config={
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "checkpoint_dir": str(tmp_path),
            "candidate_contract_mode": "candidate_id",
            "candidate_identity_mode": "topology_native",
            "experiment_config": {
                "name": "pytest_known_candidate_one_pass_drift",
                "mode": "eval_drift",
                "drift_window_size": 1,
                "drift_window_sliding": 1,
            },
        },
        prepared_data={"idx_to_version": {0: "v1"}},
    )


def test_collect_drift_inference_records_is_compact(tmp_path):
    trainer = _trainer(tmp_path)
    loader = DataLoader(
        [
            _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
            _sample(trace_idx=1, target=2, pred=1, mask=[False, True, True]),
        ],
        batch_size=2,
        shuffle=False,
    )

    records = trainer._collect_drift_inference_records(loader)

    assert records.trace_idx.tolist() == [0, 1]
    assert records.y_true.tolist() == ["class:1", "class:2"]
    assert records.y_pred.tolist() == ["class:1", "class:1"]
    assert records.fixed_y_true.tolist() == [1, 2]
    assert records.fixed_y_pred.tolist() == [1, 1]
    assert records.confidence.shape == (2,)
    assert records.target_in_mask_flags.tolist() == pytest.approx([1.0, 1.0])
    assert records.pred_in_mask_flags.tolist() == pytest.approx([1.0, 1.0])
    assert not hasattr(records, "y_prob")


def test_eval_drift_finetune_updates_after_released_window(tmp_path):
    trainer = _trainer(
        tmp_path,
        drift_window_size=2,
        drift_window_sliding=1,
        model=_TrainableThresholdModel(),
    )
    trainer.mode = "eval_drift_finetune"
    trainer.finetune_epochs = 8
    trainer.finetune_learning_rate = 0.2
    trainer.tracker = _FakeTracker()
    traces = [_trace(f"c{i}", i) for i in range(3)]
    dataset = [
        _sample(trace_idx=0, target=1, pred=0),
        _sample(trace_idx=1, target=1, pred=0),
        _sample(trace_idx=2, target=1, pred=0),
    ]
    for sample in dataset:
        sample.x_num = torch.tensor([[-0.2]], dtype=torch.float32)

    drift_metrics = trainer._evaluate_drift_finetune_windows_from_prebuilt_dataset(
        traces=traces,
        prebuilt_test_dataset=dataset,
    )

    assert drift_metrics is not None
    assert [row["finetune_update_index"] for row in drift_metrics] == [0.0, 1.0]
    assert [row["finetune_unique_traces_seen"] for row in drift_metrics] == [0.0, 1.0]
    strict_f1_by_step = [
        value
        for key, value, step in trainer.tracker.metrics
        if key == "drift_window_strict_macro_f1"
    ]
    assert strict_f1_by_step[0] < strict_f1_by_step[1]
    logged_updates = [
        value
        for key, value, step in trainer.tracker.metrics
        if key == "finetune_update_index"
    ]
    assert logged_updates == [0.0, 1.0]


def test_eval_drift_finetune_graph_indexing_emits_progress(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BPM_PROGRESS_EVENTS", "1")
    trainer = _trainer(tmp_path)

    graphs = [
        _sample(trace_idx=0, target=1, pred=0),
        _sample(trace_idx=1, target=1, pred=0),
    ]

    indexed = trainer._iter_prebuilt_graphs(graphs)

    assert indexed == graphs
    events = [
        json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(PROGRESS_EVENT_PREFIX)
    ]
    indexing_events = [
        event
        for event in events
        if event["stage"] == "eval_drift.one_pass_inference"
    ]
    assert [event["status"] for event in indexing_events] == ["start", "done"]
    assert indexing_events[0]["message"] == "Indexing adaptive drift graph stream"
    assert indexing_events[-1]["current"] == 2.0


def test_eval_drift_finetune_windows_emit_progress_before_release_cut(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BPM_PROGRESS_EVENTS", "1")
    trainer = _trainer(
        tmp_path,
        drift_window_size=2,
        drift_window_sliding=1,
        model=_TrainableThresholdModel(),
        finetune_start_ratio=1.0,
    )
    trainer.mode = "eval_drift_finetune"
    trainer.tracker = _FakeTracker()
    traces = [_trace(f"c{i}", i) for i in range(4)]
    dataset = [_sample(trace_idx=i, target=1, pred=0) for i in range(4)]
    for sample in dataset:
        sample.x_num = torch.tensor([[-0.2]], dtype=torch.float32)

    drift_metrics = trainer._evaluate_drift_finetune_windows_from_prebuilt_dataset(
        traces=traces,
        prebuilt_test_dataset=dataset,
    )

    assert drift_metrics is not None
    events = [
        json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(PROGRESS_EVENT_PREFIX)
    ]
    window_events = [
        event
        for event in events
        if event["stage"] == "eval_drift.windows"
    ]
    update_events = [event for event in window_events if event["status"] == "update"]
    assert [event["current"] for event in update_events] == [1.0, 2.0, 3.0]
    assert window_events[-1]["status"] == "done"
    assert window_events[-1]["current"] == 3.0


def test_eval_drift_finetune_start_ratio_skips_pre_cut_releases(tmp_path):
    trainer = _trainer(
        tmp_path,
        drift_window_size=2,
        drift_window_sliding=1,
        model=_TrainableThresholdModel(),
        finetune_start_ratio=0.5,
    )
    trainer.mode = "eval_drift_finetune"
    trainer.finetune_epochs = 1
    trainer.finetune_learning_rate = 0.2
    trainer.tracker = _FakeTracker()
    traces = [_trace(f"c{i}", i) for i in range(5)]
    dataset = [_sample(trace_idx=i, target=1, pred=0) for i in range(5)]
    for sample in dataset:
        sample.x_num = torch.tensor([[-0.2]], dtype=torch.float32)

    drift_metrics = trainer._evaluate_drift_finetune_windows_from_prebuilt_dataset(
        traces=traces,
        prebuilt_test_dataset=dataset,
    )

    assert drift_metrics is not None
    assert [row["finetune_start_trace"] for row in drift_metrics] == [2.0, 2.0, 2.0, 2.0]
    assert [row["finetune_update_index"] for row in drift_metrics] == [0.0, 0.0, 0.0, 1.0]
    assert [row["finetune_unique_traces_seen"] for row in drift_metrics] == [0.0, 0.0, 0.0, 1.0]
    logged_update_counts = [value for key, value, step in trainer.tracker.metrics if key == "finetune_update_index"]
    assert logged_update_counts == [0.0, 0.0, 0.0, 1.0]


def test_eval_drift_finetune_writes_sidecar_checkpoint_without_overwriting_source(tmp_path):
    trainer = _trainer(
        tmp_path,
        drift_window_size=2,
        drift_window_sliding=1,
        model=_TrainableThresholdModel(),
    )
    trainer.mode = "eval_drift_finetune"
    trainer.checkpoint_path = tmp_path / "source_best.pth"
    trainer.finetune_checkpoint_path = trainer._derive_finetune_checkpoint_path(trainer.checkpoint_path)
    trainer.checkpoint_path.write_bytes(b"source-checkpoint-sentinel")
    checkpoint = {
        "epoch": 7,
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": {},
        "val_loss": 0.25,
        "encoder_state": {"activity_vocab": {"A": 0}},
    }
    trainer._evaluate_drift_finetune_windows_from_prebuilt_dataset = lambda **kwargs: []  # type: ignore[method-assign]

    result = trainer._run_eval_drift_finetune(
        split_data=SplitData(train=[], val=[], test=[]),
        drift_traces=[],
        checkpoint=checkpoint,
        best_epoch=7,
        best_val_loss=0.25,
        prebuilt_test_dataset=[],
    )

    assert trainer.checkpoint_path.read_bytes() == b"source-checkpoint-sentinel"
    assert trainer.finetune_checkpoint_path.exists()
    assert trainer.finetune_checkpoint_path.name == "source_finetune.pth"
    assert result["finetune_checkpoint_path"] == str(trainer.finetune_checkpoint_path)


def test_eval_drift_finetune_run_does_not_enter_full_train_pipeline(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.mode = "eval_drift_finetune"
    trainer._prepare_checkpoint_state = lambda *, is_eval_mode: ({}, 0, 0.0, 0)  # type: ignore[method-assign]
    trainer._run_train_pipeline = lambda **kwargs: pytest.fail("eval_drift_finetune must not run the full train pipeline")  # type: ignore[method-assign]
    trainer._run_eval_drift_finetune = lambda **kwargs: {"mode": "eval_drift_finetune"}  # type: ignore[method-assign]

    result = trainer.run()

    assert result == {"mode": "eval_drift_finetune"}


def test_common_audit_uses_final_prediction_and_keeps_unseen_target(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.mask_guided_enabled = True
    trainer.mask_guided_policy = "hard"
    trainer.mask_guided_apply_in_eval = True
    trainer._reverse_activity_vocab = {0: "A", 1: "B", 2: "C"}
    samples = []
    for idx, (decoding, common) in enumerate([
        ([False, True, False], ["C"]), ([False, False, False], ["B"])
    ]):
        sample = _sample(trace_idx=idx, target=0, pred=1, mask=decoding)
        sample.target_label = "unseen_observed"
        sample.audit_payload_json = json.dumps({
            "audit_allowed_activity_labels": common, "audit_mask_status": "resolved",
            "audit_mask_policy_id": "reference-policy",
        })
        samples.append(sample)
    loader = DataLoader(samples, batch_size=2, shuffle=False)
    records = trainer._collect_drift_inference_records(loader)
    metrics = trainer._compute_test_metrics_from_records(records, np.array([0, 1]))
    assert records.y_pred.tolist() == ["B", "B"]
    assert records.y_true.tolist() == ["unseen_observed", "unseen_observed"]
    assert metrics["valid_prediction_count"] == 2
    assert metrics["oos_error_count"] == 1
    assert metrics["parallelism_admissible_error_count"] == 1
    assert metrics["common_oos_rate"] == 0.5
    records.version_labels = ["v3", "v5"]
    trainer.tracker = _FakeTracker()
    trainer._log_eval_drift_endpoint_audit_metrics(records)
    logged = {key: value for key, value, _ in trainer.tracker.metrics}
    assert logged["endpoint_v5_audited_prefix_count"] == 1
    assert logged["endpoint_v3_v4_v5_audited_prefix_count"] == 2


@pytest.mark.parametrize(
    ("decoding_mask", "expected_prediction", "common_labels", "expected_outcome"),
    [
        ([False, True, False], "B", ["C"], "oos_error"),
        ([False, False, False], "C", ["C"], "parallelism_admissible_error"),
    ],
)
def test_common_audit_preserves_hard_mask_and_empty_mask_fallback_prediction(
    tmp_path, decoding_mask, expected_prediction, common_labels, expected_outcome
):
    control_model = _CountingLogitModel()
    audited_model = _CountingLogitModel()
    control = _trainer(tmp_path / "control", model=control_model)
    audited = _trainer(tmp_path / "audited", model=audited_model)
    for trainer in (control, audited):
        trainer.mask_guided_enabled = True
        trainer.mask_guided_policy = "hard"
        trainer.mask_guided_apply_in_eval = True
        trainer._reverse_activity_vocab = {0: "A", 1: "B", 2: "C"}

    control_sample = _sample(trace_idx=0, target=0, pred=2, mask=decoding_mask)
    audited_sample = _sample(trace_idx=0, target=0, pred=2, mask=decoding_mask)
    audited_sample.audit_payload_json = json.dumps({
        "audit_allowed_activity_labels": common_labels,
        "audit_mask_status": "resolved",
        "audit_mask_policy_id": "reference-policy",
    })
    control_records = control._collect_drift_inference_records(
        DataLoader([control_sample], batch_size=1, shuffle=False)
    )
    audited_records = audited._collect_drift_inference_records(
        DataLoader([audited_sample], batch_size=1, shuffle=False)
    )

    assert control_model.forward_calls == audited_model.forward_calls == 1
    assert torch.equal(control_model.raw_logits[0], audited_model.raw_logits[0])
    assert control_records.y_pred.tolist() == audited_records.y_pred.tolist() == [expected_prediction]
    for field in (
        "oos_flags", "target_in_mask_flags", "pred_in_mask_flags",
        "strict_error_but_allowed_flags", "mask_cardinality",
    ):
        assert np.array_equal(getattr(control_records, field), getattr(audited_records, field))
    summary = aggregate_outcomes(audited_records.audit_observations)
    assert summary.valid_prediction_count == summary.audited_prefix_count == 1
    assert summary.excluded_count == 0
    assert classify_observation(audited_records.audit_observations[0]).outcome == expected_outcome


def test_rs01_common_audit_covers_all_evaluated_prefixes(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.mask_guided_enabled = True
    trainer.mask_guided_policy = "hard"
    trainer.mask_guided_apply_in_eval = True
    trainer._reverse_activity_vocab = {0: "A", 1: "B", 2: "C"}
    samples = []
    for trace_idx, prediction in enumerate((0, 1, 2)):
        sample = _sample(
            trace_idx=trace_idx,
            target=0,
            pred=prediction,
            mask=[False, False, False],
        )
        sample.audit_payload_json = json.dumps({
            "audit_allowed_activity_labels": ["B"],
            "audit_mask_status": "resolved",
            "audit_mask_policy_id": "reference-policy",
        })
        samples.append(sample)

    records = trainer._collect_drift_inference_records(
        DataLoader(samples, batch_size=3, shuffle=False)
    )
    metrics = trainer._compute_test_metrics_from_records(
        records,
        np.asarray([0, 1, 2], dtype=np.int64),
    )
    summary = aggregate_outcomes(records.audit_observations)

    assert summary.metric_contract_id == "state_aware_activity_label_mask.v2"
    assert metrics["valid_prediction_count"] == metrics["audited_prefix_count"] == 3
    assert metrics["excluded_count"] == 0
    assert metrics["unresolved_mapping_count"] == 0
    assert (
        metrics["strict_correct_count"]
        + metrics["parallelism_admissible_error_count"]
        + metrics["oos_error_count"]
        == metrics["valid_prediction_count"]
    )
    assert abs(
        metrics["strict_correct_rate"]
        + metrics["parallelism_admissible_error_rate"]
        + metrics["oos_error_rate"]
        - 1.0
    ) <= 1.0e-6
    assert abs(metrics["strict_correct_rate"] - metrics["strict_test_accuracy"]) <= 1.0e-6


def test_rs01_legacy_fallback_is_not_common_coverage_evidence(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.mask_guided_enabled = True
    trainer.mask_guided_policy = "hard"
    trainer.mask_guided_apply_in_eval = True
    trainer._reverse_activity_vocab = {0: "A", 1: "B", 2: "C"}
    sample = _sample(trace_idx=0, target=0, pred=1, mask=[False, False, False])

    records = trainer._collect_drift_inference_records(
        DataLoader([sample], batch_size=1, shuffle=False)
    )
    metrics = trainer._compute_test_metrics_from_records(
        records,
        np.asarray([0], dtype=np.int64),
    )
    summary = aggregate_outcomes(records.audit_observations)

    assert summary.metric_contract_id == "fixed_vocab_label_mask.v1"
    assert metrics["valid_prediction_count"] == 0
    assert metrics["excluded_count"] == 1


def test_one_pass_candidate_id_metrics_use_unseen_candidate_space(tmp_path):
    trainer = _candidate_trainer(tmp_path)
    sample = _sample(trace_idx=0, target=0, pred=0, mask=[True, False, False])
    sample.target_label = "new_task"
    loader = DataLoader([sample], batch_size=1, shuffle=False)

    records = trainer._collect_drift_inference_records(loader)
    metrics = trainer._compute_test_metrics_from_records(records, np.asarray([0], dtype=np.int64))

    assert records.y_true.tolist() == ["new_task"]
    assert records.y_pred.tolist() == ["new_task"]
    assert metrics["strict_test_accuracy"] == pytest.approx(1.0)
    assert metrics["strict_test_macro_f1"] == pytest.approx(1.0)
    assert metrics["fixed_label_strict_test_accuracy"] == pytest.approx(0.0)
    assert metrics["fixed_label_strict_test_macro_f1"] == pytest.approx(0.0)


def test_one_pass_rs01_candidate_audit_projects_fixed_mask_when_native_mask_missing(tmp_path):
    trainer = _known_candidate_trainer(tmp_path)
    sample = _sample(trace_idx=0, target=1, pred=0, mask=[False, True])
    sample.target_label = "B"
    loader = DataLoader([sample], batch_size=1, shuffle=False)

    records = trainer._collect_drift_inference_records(loader)
    metrics = trainer._compute_test_metrics_from_records(records, np.asarray([0], dtype=np.int64))

    assert metrics["audited_prefix_count"] == 1
    assert metrics["valid_prediction_count"] == 1
    assert metrics["unresolved_mapping_count"] == 0
    assert metrics["excluded_count"] == 0
    assert metrics["strict_correct_rate"] == pytest.approx(1.0)


def test_one_pass_rs01_candidate_audit_keeps_batch_native_mask_rows(tmp_path):
    trainer = _known_candidate_trainer(tmp_path)
    samples = []
    for trace_idx in range(64):
        sample = _sample(trace_idx=trace_idx, target=1, pred=0, mask=[False, True])
        sample.candidate_allowed_target_mask = torch.tensor([False, True], dtype=torch.bool)
        sample.target_label = "B"
        samples.append(sample)
    loader = DataLoader(samples, batch_size=64, shuffle=False)

    records = trainer._collect_drift_inference_records(loader)
    metrics = trainer._compute_test_metrics_from_records(records, np.arange(records.y_true.shape[0]))

    assert metrics["audited_prefix_count"] == 64
    assert metrics["valid_prediction_count"] == 64
    assert metrics["unresolved_mapping_count"] == 0
    assert metrics["excluded_count"] == 0
    assert metrics["strict_correct_rate"] == pytest.approx(1.0)


def test_one_pass_topology_native_legacy_oos_uses_native_candidate_mask(tmp_path):
    trainer = _known_candidate_trainer(tmp_path)
    sample = _sample(trace_idx=0, target=0, pred=0, mask=[True, False])
    sample.candidate_allowed_target_mask = torch.tensor([False, True], dtype=torch.bool)
    sample.target_label = "A"

    records = trainer._collect_drift_inference_records(DataLoader([sample], batch_size=1, shuffle=False))
    metrics = trainer._compute_test_metrics_from_records(records, np.asarray([0], dtype=np.int64))

    assert metrics["strict_test_accuracy"] == pytest.approx(metrics["strict_correct_rate"])
    assert metrics["test_oos"] == pytest.approx(metrics["oos_error_rate"])
    assert metrics["test_strict_error_but_allowed_rate"] == pytest.approx(
        metrics["parallelism_admissible_error_rate"]
    )
    assert metrics["strict_correct_rate"] == pytest.approx(0.0)
    assert metrics["parallelism_admissible_error_rate"] == pytest.approx(1.0)
    assert metrics["oos_error_rate"] == pytest.approx(0.0)


def test_collect_drift_inference_records_emits_one_pass_progress_events(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BPM_PROGRESS_EVENTS", "1")
    trainer = _trainer(tmp_path)
    loader = DataLoader(
        [
            _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
            _sample(trace_idx=1, target=2, pred=1, mask=[False, True, True]),
            _sample(trace_idx=2, target=0, pred=2, mask=[True, False, True]),
        ],
        batch_size=2,
        shuffle=False,
    )

    records = trainer._collect_drift_inference_records(loader)

    assert records.trace_idx.tolist() == [0, 1, 2]
    output = capsys.readouterr().out
    events = [
        json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
        for line in output.splitlines()
        if line.startswith(PROGRESS_EVENT_PREFIX)
    ]
    one_pass_events = [event for event in events if event.get("stage") == "eval_drift.one_pass_inference"]

    assert [event.get("status") for event in one_pass_events] == ["start", "update", "update", "done"]
    assert one_pass_events[-1]["current"] == 3
    assert one_pass_events[-1]["total"] == 3


def test_record_metrics_match_evaluate_test_for_full_dataset(tmp_path):
    trainer = _trainer(tmp_path)
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, False, True]),
        _sample(trace_idx=2, target=0, pred=2, mask=[True, False, True]),
    ]

    legacy_metrics = trainer._evaluate_test(DataLoader(samples, batch_size=3, shuffle=False), stage_label="eval_drift")
    records = trainer._collect_drift_inference_records(DataLoader(samples, batch_size=3, shuffle=False))
    new_metrics = trainer._compute_test_metrics_from_records(records, np.arange(records.y_true.shape[0]))

    for key in [
        "test_macro_f1",
        "strict_test_macro_f1",
        "test_accuracy",
        "strict_test_accuracy",
        "test_top3_accuracy",
        "test_oos",
        "test_target_in_mask_rate",
        "test_pred_in_mask_rate",
        "test_strict_error_but_allowed_rate",
        "test_ambiguous_prefix_rate",
    ]:
        assert new_metrics[key] == pytest.approx(legacy_metrics[key])


def test_one_pass_records_emit_rs01_outcome_partition(tmp_path):
    trainer = _trainer(tmp_path)
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, True, True]),
        _sample(trace_idx=2, target=0, pred=2, mask=[True, False, False]),
    ]

    records = trainer._collect_drift_inference_records(DataLoader(samples, batch_size=3, shuffle=False))
    metrics = trainer._compute_test_metrics_from_records(records, np.arange(records.y_true.shape[0]))

    assert metrics["strict_correct_count"] == 1
    assert metrics["parallelism_admissible_error_count"] == 1
    assert metrics["oos_error_count"] == 1
    assert metrics["valid_prediction_count"] == 3
    assert metrics["partition_sum"] == pytest.approx(1.0)
    assert metrics["strict_test_accuracy"] == pytest.approx(metrics["strict_correct_rate"])
    assert metrics["parallelism_admissible_error_rate"] == pytest.approx(
        metrics["test_strict_error_but_allowed_rate"]
    )


def test_rs01_contract_discrepancy_tolerates_float32_rounding(tmp_path):
    trainer = _trainer(tmp_path)
    strict_count = 106_474
    allowed_count = 31_185
    oos_count = 7_153
    total = strict_count + allowed_count + oos_count
    y_true = np.asarray(["A"] * total, dtype=object)
    y_pred = np.asarray(
        ["A"] * strict_count + ["B"] * allowed_count + ["B"] * oos_count,
        dtype=object,
    )
    correct = np.asarray([1.0] * strict_count + [0.0] * (allowed_count + oos_count), dtype=np.float32)
    pred_in_mask = np.asarray([1.0] * strict_count + [1.0] * allowed_count + [0.0] * oos_count, dtype=np.float32)
    target_in_mask = np.ones(total, dtype=np.float32)
    strict_error_but_allowed = np.asarray(
        [0.0] * strict_count + [1.0] * allowed_count + [0.0] * oos_count,
        dtype=np.float32,
    )
    oos_flags = np.asarray([0.0] * (strict_count + allowed_count) + [1.0] * oos_count, dtype=np.float32)
    observations = (
        tuple(
            AuditObservation("A", "A", frozenset({"A"}), "test", "test", "test.v1")
            for _ in range(strict_count)
        )
        + tuple(
            AuditObservation("B", "A", frozenset({"B"}), "test", "test", "test.v1")
            for _ in range(allowed_count)
        )
        + tuple(
            AuditObservation("B", "A", frozenset({"C"}), "test", "test", "test.v1")
            for _ in range(oos_count)
        )
    )
    records = DriftInferenceRecords(
        trace_idx=np.arange(total, dtype=np.int64),
        y_true=y_true,
        y_pred=y_pred,
        confidence=np.ones(total, dtype=np.float32),
        correct=correct,
        top3_hit=correct,
        oos_flags=oos_flags,
        target_in_mask_flags=target_in_mask,
        pred_in_mask_flags=pred_in_mask,
        strict_error_but_allowed_flags=strict_error_but_allowed,
        mask_cardinality=np.ones(total, dtype=np.float32),
        candidate_oos_flags=oos_flags,
        candidate_invalid_probability_mass=np.zeros(total, dtype=np.float32),
        candidate_valid_probability_mass=np.ones(total, dtype=np.float32),
        candidate_valid_invalid_logit_margin=np.zeros(total, dtype=np.float32),
        hybrid_correct_flags=pred_in_mask,
        hybrid_set_nll=np.zeros(total, dtype=np.float32),
        ambiguous_flags=np.ones(total, dtype=np.float32),
        prefix_lengths=np.ones(total, dtype=np.int64),
        version_labels=[],
        inference_ms_per_graph=0.0,
        fixed_y_true=np.zeros(total, dtype=np.int64),
        fixed_y_pred=np.zeros(total, dtype=np.int64),
        fixed_confidence=np.ones(total, dtype=np.float32),
        fixed_correct=correct,
        fixed_set_nll=np.zeros(total, dtype=np.float32),
        audit_observations=observations,
    )

    metrics = trainer._compute_test_metrics_from_records(records, np.arange(total, dtype=np.int64))

    assert metrics["audit_contract_discrepancy_count"] == 0.0
    assert metrics["test_strict_error_but_allowed_rate"] == pytest.approx(
        metrics["parallelism_admissible_error_rate"],
        abs=1.0e-6,
    )


def test_one_pass_rs01_exact_outside_mask_is_not_oos(tmp_path):
    trainer = _trainer(tmp_path)
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[True, False, False]),
    ]

    records = trainer._collect_drift_inference_records(DataLoader(samples, batch_size=1, shuffle=False))
    metrics = trainer._compute_test_metrics_from_records(records, np.asarray([0], dtype=np.int64))

    assert metrics["strict_correct_count"] == 1
    assert metrics["exact_outside_mask_count"] == 1
    assert metrics["oos_error_count"] == 0


def test_resolve_drift_window_record_indices_uses_trace_idx_ranges(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    samples = [
        _sample(trace_idx=0, target=1, pred=1),
        _sample(trace_idx=0, target=1, pred=1),
        _sample(trace_idx=1, target=1, pred=1),
        _sample(trace_idx=2, target=1, pred=1),
        _sample(trace_idx=2, target=1, pred=1),
        _sample(trace_idx=3, target=1, pred=1),
    ]
    records = trainer._collect_drift_inference_records(DataLoader(samples, batch_size=6, shuffle=False))
    traces = [_trace(f"c{idx}", idx) for idx in range(4)]

    windows = trainer._resolve_drift_window_record_indices(records, traces)

    assert [item[1].tolist() for item in windows] == [[0, 1, 2], [2, 3, 4], [3, 4, 5]]


def test_run_eval_drift_uses_one_pass_prebuilt_dataset_without_legacy_graph_rebuild(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, False, True]),
        _sample(trace_idx=2, target=0, pred=2, mask=[True, False, True]),
        _sample(trace_idx=3, target=1, pred=1, mask=[False, True, False]),
    ]
    traces = [_trace(f"c{idx}", idx) for idx in range(4)]

    def _fail_legacy_build_loader(window_traces, shuffle):
        _ = window_traces
        _ = shuffle
        raise AssertionError("legacy graph rebuild path should not be called")

    trainer._build_loader = _fail_legacy_build_loader  # type: ignore[method-assign]

    result = trainer._run_eval_drift(
        split_data=type("Split", (), {"train": [], "val": [], "test": traces})(),
        drift_traces=traces,
        best_epoch=1,
        best_val_loss=0.5,
        prebuilt_test_dataset=samples,
    )

    assert result["mode"] == "eval_drift"
    assert len(result["drift_metrics"]) == 3


def test_run_eval_drift_falls_back_when_prebuilt_dataset_lacks_trace_idx(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    samples = [Data(y=torch.tensor([1], dtype=torch.long))]
    traces = [_trace(f"c{idx}", idx) for idx in range(2)]

    trainer._evaluate_drift_windows = lambda drift_traces: [{"window_index": 0.0}]  # type: ignore[method-assign]

    result = trainer._run_eval_drift(
        split_data=type("Split", (), {"train": [], "val": [], "test": traces})(),
        drift_traces=traces,
        best_epoch=1,
        best_val_loss=0.5,
        prebuilt_test_dataset=samples,
    )

    assert result["drift_metrics"] == [{"window_index": 0.0}]


def test_one_pass_drift_rows_preserve_legacy_output_keys(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, False, True]),
    ]
    traces = [_trace(f"c{idx}", idx) for idx in range(2)]

    rows = trainer._evaluate_drift_windows_from_prebuilt_dataset(traces, samples)

    assert rows is not None
    assert rows
    expected_keys = {
        "window_index",
        "window_start_trace",
        "window_end_trace",
        "window_start_ts",
        "window_end_ts",
        "window_macro_f1",
        "window_strict_macro_f1",
        "window_test_ece",
        "window_test_set_nll",
        "window_test_oos",
        "window_oos_confidence_mean",
        "window_target_in_mask_rate",
        "window_pred_in_mask_rate",
        "window_strict_error_but_allowed_rate",
        "window_parallelism_admissible_error_rate",
        "window_oos_error_rate",
        "window_partition_sum",
        "window_ambiguous_prefix_rate",
    }
    assert expected_keys.issubset(rows[0].keys())
    assert rows[0]["window_oos_confidence_mean"] >= 0.0


def test_one_pass_drift_logs_legacy_tracker_metric_names(tmp_path):
    tracker = _FakeTracker()
    trainer = _trainer(tmp_path, drift_window_size=2, drift_window_sliding=1)
    trainer.tracker = tracker
    samples = [
        _sample(trace_idx=0, target=1, pred=1, mask=[False, True, False]),
        _sample(trace_idx=1, target=2, pred=1, mask=[False, False, True]),
    ]
    traces = [_trace(f"c{idx}", idx) for idx in range(2)]

    rows = trainer._evaluate_drift_windows_from_prebuilt_dataset(traces, samples)

    assert rows is not None
    logged_names = {name for name, _, _ in tracker.metrics}
    assert {
        "drift_window_macro_f1",
        "drift_window_strict_macro_f1",
        "drift_window_test_ece",
        "drift_window_test_set_nll",
        "drift_window_test_oos",
        "drift_window_oos_confidence_mean",
        "drift_window_target_in_mask_rate",
        "drift_window_pred_in_mask_rate",
        "drift_window_strict_error_but_allowed_rate",
        "drift_window_parallelism_admissible_error_rate",
        "drift_window_oos_error_rate",
        "drift_window_partition_sum",
        "drift_window_ambiguous_prefix_rate",
        "drift_window_start_ts",
        "drift_window_end_ts",
        "strict_correct_rate",
        "parallelism_admissible_error_rate",
        "oos_error_rate",
        "audited_prefix_count",
        "valid_prediction_count",
        "excluded_count",
        "unresolved_mapping_count",
        "strict_test_macro_f1",
        "strict_test_accuracy",
        "test_oos",
    }.issubset(logged_names)


def test_one_pass_fixed_head_records_use_stable_target_labels_for_future_activity(tmp_path):
    trainer = _trainer(tmp_path, drift_window_size=1, drift_window_sliding=1)
    trainer._reverse_activity_vocab = {0: "<UNK>", 1: "known_task", 2: "other_task"}
    sample = _sample(trace_idx=0, target=0, pred=0, mask=[True, False, False])
    sample.target_label = "new_future_task"

    records = trainer._collect_drift_inference_records(DataLoader([sample], batch_size=1))
    metrics = trainer._compute_test_metrics_from_records(records, np.asarray([0], dtype=np.int64))

    assert records.y_true.tolist() == ["new_future_task"]
    assert records.y_pred.tolist() == ["<UNK>"]
    assert metrics["strict_test_accuracy"] == pytest.approx(0.0)
    assert metrics["test_ece"] > 0.99
    assert metrics["test_set_nll"] > 20.0
    assert metrics["fixed_label_strict_test_accuracy"] == pytest.approx(1.0)
    assert metrics["fixed_label_test_ece"] < 0.01
    assert metrics["fixed_label_test_set_nll"] < 0.01
