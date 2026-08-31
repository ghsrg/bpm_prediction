from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from src.application.use_cases.topology_mask_uniform_evaluator import TopologyMaskUniformEvaluator
from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.uniform_mask_scorer import UniformMaskScorer
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository


def _native_data(mask, target_label):
    return Data(
        y=torch.tensor([0]),
        target_label=target_label,
        candidate_allowed_target_mask=torch.tensor(mask, dtype=torch.bool),
        candidate_ids=("node_b", "node_a", "node_c"),
        candidate_labels=("B", "A", "A"),
        prefix_len=torch.tensor([2]),
        process_version_idx=torch.tensor([2]),
        trace_idx=torch.tensor([0]),
        prefix_idx=torch.tensor([0]),
        trace_start_ts=torch.tensor([10.0]),
        trace_end_ts=torch.tensor([20.0]),
    )


def test_native_evaluator_uses_exact_candidate_mask_and_duplicate_target_set():
    result = TopologyMaskUniformEvaluator().evaluate(
        [
            _native_data([True, True, True], "A"),
            _native_data([True, False, False], "A"),
        ]
    )
    metrics = result["test_metrics"]
    assert metrics["test_pred_in_mask_rate"] == pytest.approx(1.0)
    assert metrics["test_oos"] == pytest.approx(0.0)
    assert metrics["test_target_in_mask_rate"] == pytest.approx(0.5)
    assert metrics["ranking_eligible_count"] == 1
    assert metrics["mask_failure_count"] == 1
    assert metrics["uniform_mask_expected_accuracy"] == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize(
    ("target_label", "expected_accuracy", "target_in_mask"),
    [("B", 1.0, 1.0), ("D", 0.0, 0.0)],
)
def test_native_evaluator_explains_singleton_mask(target_label, expected_accuracy, target_in_mask):
    result = TopologyMaskUniformEvaluator().evaluate(
        [
            _native_data([True, False, False], target_label),
        ]
    )
    metrics = result["test_metrics"]
    assert metrics["uniform_mask_expected_accuracy"] == pytest.approx(expected_accuracy)
    assert metrics["test_target_in_mask_rate"] == pytest.approx(target_in_mask)
    assert metrics["mask_card_1_rate"] == pytest.approx(1.0)
    assert metrics["strict_test_macro_f1_mc_mean"] == pytest.approx(expected_accuracy)
    assert metrics["strict_test_macro_f1_mc_std"] == pytest.approx(0.0)


def test_native_evaluator_assigns_one_third_to_target_inside_three_candidate_mask():
    data = _native_data([True, True, True], "C")
    data.candidate_labels = ("B", "C", "D")
    result = TopologyMaskUniformEvaluator().evaluate([data])
    assert result["test_metrics"]["uniform_mask_expected_accuracy"] == pytest.approx(1.0 / 3.0)


def test_target_outside_mask_is_reported_as_mask_failure_not_ranking_eligible():
    result = TopologyMaskUniformEvaluator().evaluate(
        [
            _native_data([True, True, False], "D"),
        ]
    )
    metrics = result["test_metrics"]
    assert metrics["test_target_in_mask_rate"] == pytest.approx(0.0)
    assert metrics["uniform_mask_expected_accuracy"] == pytest.approx(0.0)
    assert metrics["mask_failure_rate"] == pytest.approx(1.0)
    assert metrics["mask_failure_count"] == 1
    assert metrics["ranking_eligible_rate"] == pytest.approx(0.0)
    assert metrics["ranking_eligible_count"] == 0


def test_native_evaluator_emits_cardinality_diagnostics_and_empty_rate():
    result = TopologyMaskUniformEvaluator(empty_mask_policy="record_invalid").evaluate(
        [
            _native_data([True, False, False], "B"),
            _native_data([True, True, False], "A"),
            _native_data([True, True, True], "A"),
            _native_data([False, False, False], "A"),
        ]
    )
    metrics = result["test_metrics"]
    assert metrics["empty_mask_rate"] == pytest.approx(0.25)
    assert metrics["mask_card_1_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["mean_mask_cardinality"] == pytest.approx(2.0)
    assert metrics["median_mask_cardinality"] == pytest.approx(2.0)
    assert "strict_test_macro_f1_mc_mean_mask_card_1" in metrics
    assert "strict_test_macro_f1_mc_mean_mask_card_2" in metrics
    assert "strict_test_macro_f1_mc_mean_mask_card_3_plus" in metrics
    assert metrics["mask_card_1_count"] == 1
    assert metrics["mask_card_2_count"] == 1
    assert metrics["mask_card_3_plus_count"] == 1
    assert metrics["candidate_reduction_ratio_mean"] == pytest.approx(1.0 / 3.0)
    assert metrics["candidate_reduction_ratio_median"] == pytest.approx(1.0 / 3.0)


def test_monte_carlo_draw_count_must_be_in_research_reporting_range():
    with pytest.raises(ValueError, match="mc_draws must be between 100 and 1000\\."):
        TopologyMaskUniformEvaluator(mc_draws=99)


def test_native_evaluator_keeps_unseen_future_candidate_representable():
    data = _native_data([False, True, True], "X")
    data.candidate_ids = ("node_A", "node_B", "node_X")
    data.candidate_labels = ("A", "B", "X")
    data.candidate_class_index = torch.tensor([0, 1, -1])
    result = TopologyMaskUniformEvaluator().evaluate([data])
    metrics = result["test_metrics"]
    assert metrics["uniform_mask_expected_accuracy"] == pytest.approx(0.5)
    assert metrics["test_target_in_mask_rate"] == pytest.approx(1.0)


def test_order_permutation_keeps_probability_and_seeded_draw_by_candidate_id():
    scorer = UniformMaskScorer()
    first = scorer.score(
        allowed_mask=torch.tensor([[True, False, True]]),
        candidate_keys=("node_B", "node_C", "node_A"),
    )
    second = scorer.score(
        allowed_mask=torch.tensor([[True, True, False]]),
        candidate_keys=("node_A", "node_B", "node_C"),
    )
    first_by_id = dict(zip(("node_B", "node_C", "node_A"), first.probabilities[0].tolist()))
    second_by_id = dict(zip(("node_A", "node_B", "node_C"), second.probabilities[0].tolist()))
    assert first_by_id == second_by_id
    first_idx = scorer.sample_prediction(
        allowed_mask=torch.tensor([[True, False, True]]),
        candidate_keys=("node_B", "node_C", "node_A"),
        evaluation_seed=41,
        draw_index=7,
        sample_key="v2/trace-4/prefix-2",
    )
    second_idx = scorer.sample_prediction(
        allowed_mask=torch.tensor([[True, True, False]]),
        candidate_keys=("node_A", "node_B", "node_C"),
        evaluation_seed=41,
        draw_index=7,
        sample_key="v2/trace-4/prefix-2",
    )
    assert ("node_B", "node_C", "node_A")[first_idx] == ("node_A", "node_B", "node_C")[second_idx]


def test_monte_carlo_summary_is_reproducible_and_reports_conditional_interval():
    result = TopologyMaskUniformEvaluator(evaluation_seed=41, mc_draws=100).evaluate(
        [
            _native_data([True, False, False], "B"),
        ]
    )
    metrics = result["test_metrics"]
    assert metrics["strict_test_macro_f1_mc_mean"] == pytest.approx(1.0)
    assert metrics["strict_test_macro_f1_mc_std"] == pytest.approx(0.0)
    assert metrics["strict_test_macro_f1_mc_sampling_uncertainty_95_low"] == pytest.approx(1.0)
    assert metrics["strict_test_macro_f1_mc_sampling_uncertainty_95_high"] == pytest.approx(1.0)
    assert result["monte_carlo"]["evaluation_seed"] == 41
    assert result["monte_carlo"]["draws"] == 100
    assert result["monte_carlo"]["interval_label"] == "Monte Carlo sampling uncertainty interval (95%)"


def test_strict_error_but_allowed_rate_is_reported_per_draw_for_uniform_misses():
    result = TopologyMaskUniformEvaluator(evaluation_seed=41, mc_draws=100).evaluate(
        [
            _native_data([True, True, False], "A"),
        ]
    )
    metrics = result["test_metrics"]
    assert metrics["uniform_mask_expected_accuracy"] == pytest.approx(0.5)
    assert metrics["strict_error_but_allowed_rate_mc_mean"] > 0.0
    assert metrics["strict_error_but_allowed_rate_mc_mean"] < 1.0
    assert metrics["strict_error_but_allowed_rate"] == pytest.approx(
        metrics["strict_error_but_allowed_rate_mc_mean"]
    )
    assert metrics["test_strict_error_but_allowed_rate"] == pytest.approx(
        metrics["strict_error_but_allowed_rate_mc_mean"]
    )


def test_mou_hybrid_macro_f1_uses_model_trainer_ambiguous_mask_policy():
    result = TopologyMaskUniformEvaluator(evaluation_seed=41, mc_draws=100).evaluate(
        [
            _native_data([True, True, False], "A"),
        ]
    )
    metrics = result["test_metrics"]
    assert metrics["strict_test_macro_f1_mc_mean"] < 1.0
    assert metrics["test_macro_f1_mc_mean"] == pytest.approx(1.0)
    assert metrics["legacy_test_macro_f1_mc_mean"] == pytest.approx(
        metrics["strict_test_macro_f1_mc_mean"]
    )


def test_prepared_sample_preserves_builder_native_mask_for_mou(mock_feature_configs):
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", ["A", "B"])],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v2",
        ProcessStructureDTO(
            version="v2",
            allowed_edges=[("A", "B")],
            nodes=[
                {"id": "node_A", "bpmn_tag": "task", "type": "task", "label": "A"},
                {"id": "node_B", "bpmn_tag": "task", "type": "task", "label": "B"},
            ],
        ),
    )
    prefix = PrefixSlice(
        case_id="eval",
        process_version="v2",
        prefix_events=[_event(0, "A")],
        target_event=_event(1, "B"),
    )
    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        candidate_identity_mode="topology_native",
    ).build_graph(prefix)
    sample = Data(
        y=contract["y"],
        target_label=contract["target_label"],
        candidate_allowed_target_mask=contract["candidate_allowed_target_mask"].unsqueeze(0),
        candidate_ids=contract["candidate_ids"],
        candidate_labels=contract["candidate_labels"],
        prefix_len=torch.tensor([1]),
        process_version_idx=torch.tensor([2]),
        trace_idx=torch.tensor([0]),
        prefix_idx=torch.tensor([0]),
        trace_start_ts=torch.tensor([10.0]),
        trace_end_ts=torch.tensor([20.0]),
    )
    record = TopologyMaskUniformEvaluator().evaluate_sample(sample)
    assert record.candidate_ids == tuple(contract["candidate_ids"])
    assert record.allowed_mask == tuple(contract["candidate_allowed_target_mask"].tolist())
    assert record.mask_cardinality == int(contract["candidate_allowed_target_mask"].sum())


def test_prebuilt_trace_metadata_produces_scalar_drift_windows():
    first = _native_data([True, False, False], "B")
    second = _native_data([True, True, False], "A")
    second.trace_idx = torch.tensor([1])
    result = TopologyMaskUniformEvaluator(drift_window_size=1).evaluate([first, second])
    assert len(result["drift_metrics"]) == 2
    assert all("window_uniform_mask_expected_accuracy" in row for row in result["drift_metrics"])
    assert all("window_strict_test_macro_f1_mc_mean" in row for row in result["drift_metrics"])


def test_drift_windows_use_eval_drift_sliding_step_and_drop_short_tail():
    samples = []
    for idx in range(12):
        for prefix_idx in range(2):
            sample = _native_data([True, True, False], "A")
            sample.trace_idx = torch.tensor([idx])
            sample.prefix_idx = torch.tensor([prefix_idx])
            sample.trace_start_ts = torch.tensor([float(idx)])
            sample.trace_end_ts = torch.tensor([float(idx + 1)])
            samples.append(sample)

    result = TopologyMaskUniformEvaluator(
        evaluation_seed=41,
        mc_draws=100,
        drift_window_size=5,
        drift_window_sliding=2,
    ).evaluate(samples)

    rows = result["drift_metrics"]
    assert [row["window_start_trace_idx"] for row in rows] == [0, 2, 4, 6]
    assert [row["window_count"] for row in rows] == [10, 10, 10, 10]
    assert all("window_macro_f1" in row for row in rows)
    assert all("window_strict_error_but_allowed_rate" in row for row in rows)
    assert all("window_test_oos" in row for row in rows)


def test_evaluator_reports_eval_drift_progress_stages_only():
    events = []

    def _record_progress(**event):
        events.append(event)

    TopologyMaskUniformEvaluator(
        evaluation_seed=41,
        mc_draws=100,
        drift_window_size=1,
        progress_callback=_record_progress,
    ).evaluate([_native_data([True, False, False], "B"), _native_data([True, True, False], "A")])

    seen = {(event["stage"], event["status"]) for event in events}
    assert {event["stage"] for event in events} <= {"eval_drift.one_pass_inference", "eval_drift.windows"}
    assert ("eval_drift.one_pass_inference", "start") in seen
    assert ("eval_drift.one_pass_inference", "done") in seen
    assert ("eval_drift.windows", "start") in seen
    assert ("eval_drift.windows", "done") in seen
    assert any(
        event["stage"] == "eval_drift.one_pass_inference"
        and event["status"] == "update"
        and event["current"] == 102
        and event["total"] == 102
        for event in events
    )


def test_monte_carlo_predictions_are_reused_across_windows_and_cardinality():
    samples = []
    for idx in range(6):
        sample = _native_data([True, True, False], "A")
        sample.trace_idx = torch.tensor([idx])
        sample.trace_start_ts = torch.tensor([float(idx)])
        sample.trace_end_ts = torch.tensor([float(idx + 1)])
        samples.append(sample)

    evaluator = TopologyMaskUniformEvaluator(
        evaluation_seed=41,
        mc_draws=100,
        drift_window_size=3,
        drift_window_sliding=1,
    )
    original = evaluator.scorer.sample_prediction
    calls = 0

    def _counting_sample_prediction(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    evaluator.scorer.sample_prediction = _counting_sample_prediction
    evaluator.evaluate(samples)

    assert calls == len(samples) * evaluator.mc_draws


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
