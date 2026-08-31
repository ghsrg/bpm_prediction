from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from statistics import mean, median
from typing import Any, Callable, Iterable, Sequence

import torch

from src.application.services.candidate_target_mapping import candidate_target_mask_from_labels
from src.domain.services.candidate_label_matching import candidate_label_metric_key
from src.domain.services.uniform_mask_scorer import UniformMaskScorer


@dataclass(frozen=True)
class UniformMaskEvaluationRecord:
    sample_key: str
    candidate_ids: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    allowed_mask: tuple[bool, ...]
    target_mask: tuple[bool, ...]
    target_label: str
    mask_cardinality: int
    target_match_count: int
    target_in_mask: bool
    expected_accuracy: float
    reduction_ratio: float
    invalid: bool
    trace_idx: int
    trace_start_ts: float
    trace_end_ts: float


class TopologyMaskUniformEvaluator:
    def __init__(
        self,
        *,
        empty_mask_policy: str = "raise",
        evaluation_seed: int = 20260831,
        mc_draws: int = 200,
        drift_window_size: int = 0,
        drift_window_sliding: int = 0,
        progress_callback: Callable[..., None] | None = None,
    ) -> None:
        if mc_draws < 100 or mc_draws > 1000:
            raise ValueError("mc_draws must be between 100 and 1000.")
        if drift_window_sliding < 0:
            raise ValueError("drift_window_sliding must be non-negative.")
        self.empty_mask_policy = empty_mask_policy
        self.evaluation_seed = int(evaluation_seed)
        self.mc_draws = int(mc_draws)
        self.drift_window_size = int(drift_window_size or 0)
        self.drift_window_sliding = int(drift_window_sliding or 0)
        self.progress_callback = progress_callback
        self.scorer = UniformMaskScorer(empty_mask_policy=empty_mask_policy)
        self._prediction_cache: dict[tuple[int, str], int] = {}

    def evaluate(self, samples: Iterable[Any]) -> dict[str, Any]:
        self._prediction_cache.clear()
        records: list[UniformMaskEvaluationRecord] = []
        sample_total = len(samples) if hasattr(samples, "__len__") else None
        one_pass_total = (int(sample_total) if sample_total is not None else 0) + self.mc_draws
        self._emit_progress(
            stage="eval_drift.one_pass_inference",
            status="start",
            message="Running one-pass topology mask inference",
            current=0,
            total=one_pass_total if sample_total is not None else None,
        )
        for idx, sample in enumerate(samples, start=1):
            records.append(self.evaluate_sample(sample))
            self._emit_progress(
                stage="eval_drift.one_pass_inference",
                status="update",
                message="Running one-pass topology mask inference",
                current=idx,
                total=one_pass_total if sample_total is not None else None,
                payload={"records": idx},
            )

        draw_metrics = []
        for draw_index in range(self.mc_draws):
            draw_metrics.append(self._aggregate_draw_metrics(records, draw_index))
            self._emit_progress(
                stage="eval_drift.one_pass_inference",
                status="update",
                message="Running one-pass topology mask inference",
                current=len(records) + draw_index + 1,
                total=len(records) + self.mc_draws,
                payload={"records": len(records), "draws": self.mc_draws},
            )

        metrics = self._aggregate_analytical_metrics(records)
        metrics.update(self._summarize_monte_carlo(draw_metrics))
        self._append_cardinality_metrics(metrics, records, draw_metrics)
        self._emit_progress(
            stage="eval_drift.one_pass_inference",
            status="done",
            message="One-pass topology mask inference completed",
            current=len(records) + self.mc_draws,
            total=len(records) + self.mc_draws,
            payload={"records": len(records), "draws": self.mc_draws},
        )
        drift_metrics = self._drift_metrics(records) if self.drift_window_size > 0 else []
        return {
            "test_metrics": metrics,
            "monte_carlo": {
                "evaluation_seed": self.evaluation_seed,
                "draws": self.mc_draws,
                "interval_label": "Monte Carlo sampling uncertainty interval (95%)",
            },
            "drift_metrics": drift_metrics,
        }

    def evaluate_sample(self, data: Any) -> UniformMaskEvaluationRecord:
        payload = self._resolve_native_payload(data)
        score = self.scorer.score(
            allowed_mask=payload["allowed_mask"].unsqueeze(0),
            candidate_keys=payload["candidate_ids"],
        )
        target_mask = candidate_target_mask_from_labels(
            target_labels=[payload["target_label"]],
            candidate_labels=payload["candidate_labels"],
            candidate_ids=payload["candidate_ids"],
            device=payload["allowed_mask"].device,
        )[0]
        allowed_mask = payload["allowed_mask"].to(dtype=torch.bool)
        target_matches = allowed_mask & target_mask
        cardinality = int(score.mask_cardinality[0].detach().cpu().item())
        invalid = bool(score.invalid_rows[0].detach().cpu().item())
        target_match_count = int(target_matches.sum().detach().cpu().item()) if not invalid else 0
        expected_accuracy = float(target_match_count / cardinality) if cardinality > 0 else 0.0
        candidate_count = max(1, len(payload["candidate_ids"]))
        reduction_ratio = 1.0 - (float(cardinality) / float(candidate_count)) if not invalid else 0.0
        return UniformMaskEvaluationRecord(
            sample_key=payload["sample_key"],
            candidate_ids=tuple(payload["candidate_ids"]),
            candidate_labels=tuple(payload["candidate_labels"]),
            allowed_mask=tuple(bool(item) for item in allowed_mask.detach().cpu().tolist()),
            target_mask=tuple(bool(item) for item in target_mask.detach().cpu().tolist()),
            target_label=payload["target_label"],
            mask_cardinality=cardinality,
            target_match_count=target_match_count,
            target_in_mask=target_match_count > 0,
            expected_accuracy=expected_accuracy,
            reduction_ratio=reduction_ratio,
            invalid=invalid,
            trace_idx=payload["trace_idx"],
            trace_start_ts=payload["trace_start_ts"],
            trace_end_ts=payload["trace_end_ts"],
        )

    def _resolve_native_payload(self, data: Any) -> dict[str, Any]:
        allowed = getattr(data, "candidate_allowed_target_mask", None)
        if not isinstance(allowed, torch.Tensor):
            raise ValueError("candidate_allowed_target_mask is required for topology mask uniform evaluation.")
        allowed = allowed.to(dtype=torch.bool)
        if allowed.dim() == 2:
            if allowed.size(0) != 1:
                raise ValueError("evaluate_sample requires one candidate mask row.")
            allowed = allowed[0]
        if allowed.dim() != 1:
            raise ValueError("candidate_allowed_target_mask must have one native candidate row.")
        candidate_ids = tuple(str(item) for item in getattr(data, "candidate_ids", ()))
        candidate_labels = tuple(str(item) for item in getattr(data, "candidate_labels", ()))
        if len(candidate_ids) != int(allowed.numel()) or len(candidate_labels) != int(allowed.numel()):
            raise ValueError("candidate ids/labels must match candidate_allowed_target_mask width.")
        target_label = str(getattr(data, "target_label", "")).strip()
        if not target_label:
            raise ValueError("target_label is required for topology mask uniform evaluation.")
        process_version_idx = self._scalar_int(getattr(data, "process_version_idx", 0))
        trace_idx = self._scalar_int(getattr(data, "trace_idx", 0))
        prefix_idx = self._scalar_int(getattr(data, "prefix_idx", 0))
        return {
            "allowed_mask": allowed,
            "candidate_ids": candidate_ids,
            "candidate_labels": candidate_labels,
            "target_label": target_label,
            "sample_key": f"{process_version_idx}/{trace_idx}/{prefix_idx}",
            "trace_idx": trace_idx,
            "trace_start_ts": self._scalar_float(getattr(data, "trace_start_ts", 0.0)),
            "trace_end_ts": self._scalar_float(getattr(data, "trace_end_ts", 0.0)),
        }

    def _aggregate_analytical_metrics(self, records: Sequence[UniformMaskEvaluationRecord]) -> dict[str, float | int]:
        total = len(records)
        valid = [record for record in records if not record.invalid]
        valid_count = len(valid)
        target_in_mask = [record for record in valid if record.target_in_mask]
        mask_failures = [record for record in valid if not record.target_in_mask]
        return {
            "uniform_mask_expected_accuracy": mean([record.expected_accuracy for record in valid]) if valid else 0.0,
            "test_target_in_mask_rate": len(target_in_mask) / valid_count if valid_count else 0.0,
            "target_in_mask_rate": len(target_in_mask) / valid_count if valid_count else 0.0,
            "test_pred_in_mask_rate": 1.0 if valid_count else 0.0,
            "test_oos": 0.0,
            "ranking_eligible_count": len(target_in_mask),
            "ranking_eligible_rate": len(target_in_mask) / valid_count if valid_count else 0.0,
            "mask_failure_count": len(mask_failures),
            "mask_failure_rate": len(mask_failures) / valid_count if valid_count else 0.0,
            "empty_mask_count": total - valid_count,
            "empty_mask_rate": (total - valid_count) / total if total else 0.0,
        }

    def _aggregate_draw_metrics(
        self,
        records: Sequence[UniformMaskEvaluationRecord],
        draw_index: int,
    ) -> dict[str, float]:
        valid = [record for record in records if not record.invalid]
        strict_true: list[str] = []
        strict_pred: list[str] = []
        hybrid_true: list[str] = []
        hybrid_pred: list[str] = []
        eligible_true: list[str] = []
        eligible_pred: list[str] = []
        strict_correct = []
        strict_error_but_allowed = []
        for record in valid:
            pred_idx = self._sample_prediction(record, draw_index)
            pred_label = record.candidate_labels[pred_idx] if pred_idx >= 0 else "__invalid_candidate_prediction__"
            true_key = candidate_label_metric_key(record.target_label)
            pred_key = candidate_label_metric_key(pred_label)
            strict_true.append(true_key)
            strict_pred.append(pred_key)
            is_correct = pred_idx >= 0 and bool(record.target_mask[pred_idx])
            strict_correct.append(1.0 if is_correct else 0.0)
            pred_in_mask = pred_idx >= 0 and bool(record.allowed_mask[pred_idx])
            strict_error_but_allowed.append(1.0 if (not is_correct and pred_in_mask) else 0.0)
            hybrid_true.append(pred_key if record.mask_cardinality > 1 and pred_in_mask else true_key)
            hybrid_pred.append(pred_key)
            if record.target_in_mask:
                eligible_true.append(true_key)
                eligible_pred.append(pred_key)
        return {
            "strict_test_macro_f1": self._macro_f1(strict_true, strict_pred),
            "test_macro_f1": self._macro_f1(hybrid_true, hybrid_pred),
            "legacy_test_macro_f1": self._macro_f1(strict_true, strict_pred),
            "ranking_eligible_macro_f1": self._macro_f1(eligible_true, eligible_pred),
            "strict_test_accuracy": mean(strict_correct) if strict_correct else 0.0,
            "strict_error_but_allowed_rate": mean(strict_error_but_allowed)
            if strict_error_but_allowed
            else 0.0,
        }

    def _sample_prediction(self, record: UniformMaskEvaluationRecord, draw_index: int) -> int:
        key = (int(draw_index), record.sample_key)
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return cached
        pred_idx = self.scorer.sample_prediction(
            allowed_mask=torch.tensor([record.allowed_mask], dtype=torch.bool),
            candidate_keys=record.candidate_ids,
            evaluation_seed=self.evaluation_seed,
            draw_index=draw_index,
            sample_key=record.sample_key,
        )
        self._prediction_cache[key] = pred_idx
        return pred_idx

    def _summarize_monte_carlo(self, draw_metrics: Sequence[dict[str, float]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for source_key, prefix in (
            ("strict_test_macro_f1", "strict_test_macro_f1"),
            ("test_macro_f1", "test_macro_f1"),
            ("legacy_test_macro_f1", "legacy_test_macro_f1"),
            ("ranking_eligible_macro_f1", "ranking_eligible_macro_f1"),
            ("strict_error_but_allowed_rate", "strict_error_but_allowed_rate"),
        ):
            values = [row[source_key] for row in draw_metrics]
            out.update(self._summary(prefix, values))
        out["strict_error_but_allowed_rate"] = out["strict_error_but_allowed_rate_mc_mean"]
        out["test_strict_error_but_allowed_rate"] = out["strict_error_but_allowed_rate_mc_mean"]
        return out

    def _append_cardinality_metrics(
        self,
        metrics: dict[str, float | int],
        records: Sequence[UniformMaskEvaluationRecord],
        draw_metrics: Sequence[dict[str, float]],
    ) -> None:
        valid = [record for record in records if not record.invalid]
        denom = len(valid)
        cards = [record.mask_cardinality for record in valid]
        reductions = [record.reduction_ratio for record in valid]
        metrics["mean_mask_cardinality"] = mean(cards) if cards else 0.0
        metrics["median_mask_cardinality"] = median(cards) if cards else 0.0
        metrics["candidate_reduction_ratio_mean"] = mean(reductions) if reductions else 0.0
        metrics["candidate_reduction_ratio_median"] = median(reductions) if reductions else 0.0
        buckets = (
            ("1", [record for record in valid if record.mask_cardinality == 1]),
            ("2", [record for record in valid if record.mask_cardinality == 2]),
            ("3_plus", [record for record in valid if record.mask_cardinality >= 3]),
        )
        for bucket, records_in_bucket in buckets:
            metrics[f"mask_card_{bucket}_count"] = len(records_in_bucket)
            metrics[f"mask_card_{bucket}_rate"] = len(records_in_bucket) / denom if denom else 0.0
            values = []
            for draw_index in range(self.mc_draws):
                values.append(self._aggregate_draw_metrics(records_in_bucket, draw_index)["strict_test_macro_f1"])
            metrics.update(self._summary(f"strict_test_macro_f1_mask_card_{bucket}", values))
            bucket_summary = self._summary("strict_test_macro_f1", values)
            for key, value in bucket_summary.items():
                metrics[f"{key}_mask_card_{bucket}"] = value

    def _drift_metrics(self, records: Sequence[UniformMaskEvaluationRecord]) -> list[dict[str, float | int]]:
        ordered = sorted(records, key=lambda item: (item.trace_start_ts, item.trace_idx, item.sample_key))
        rows: list[dict[str, float | int]] = []
        step = self._resolve_drift_step()
        trace_order = self._ordered_trace_indices(ordered)
        by_trace: dict[int, list[UniformMaskEvaluationRecord]] = {}
        for record in ordered:
            by_trace.setdefault(record.trace_idx, []).append(record)
        windows: list[tuple[int, list[UniformMaskEvaluationRecord], list[int]]] = []
        for start in range(0, len(trace_order), step):
            trace_window = trace_order[start : start + self.drift_window_size]
            if len(trace_window) < self.drift_window_size:
                continue
            window = [record for trace_idx in trace_window for record in by_trace.get(trace_idx, [])]
            windows.append((start, window, trace_window))
        self._emit_progress(
            stage="eval_drift.windows",
            status="start",
            message="Evaluating drift windows",
            current=0,
            total=len(windows),
            payload={"windows": len(windows), "draws": self.mc_draws},
        )
        for start, window, trace_window in windows:
            if not window:
                continue
            draw_metrics = [self._aggregate_draw_metrics(window, draw_index) for draw_index in range(self.mc_draws)]
            strict = self._summary("window_strict_test_macro_f1", [row["strict_test_macro_f1"] for row in draw_metrics])
            hybrid = self._summary("window_test_macro_f1", [row["test_macro_f1"] for row in draw_metrics])
            strict_allowed = self._summary(
                "window_strict_error_but_allowed_rate",
                [row["strict_error_but_allowed_rate"] for row in draw_metrics],
            )
            analytical = self._aggregate_analytical_metrics(window)
            rows.append(
                {
                    "window_index": len(rows),
                    "window_start_trace": start,
                    "window_start_trace_idx": trace_window[0],
                    "window_end_trace_idx": trace_window[-1],
                    "window_count": len(window),
                    "window_macro_f1": hybrid["window_test_macro_f1_mc_mean"],
                    "window_strict_macro_f1": strict["window_strict_test_macro_f1_mc_mean"],
                    "window_test_oos": float(analytical["test_oos"]),
                    "window_target_in_mask_rate": float(analytical["test_target_in_mask_rate"]),
                    "window_pred_in_mask_rate": float(analytical["test_pred_in_mask_rate"]),
                    "window_strict_error_but_allowed_rate": strict_allowed[
                        "window_strict_error_but_allowed_rate_mc_mean"
                    ],
                    "window_uniform_mask_expected_accuracy": mean(
                        [record.expected_accuracy for record in window if not record.invalid]
                    )
                    if any(not record.invalid for record in window)
                    else 0.0,
                    **strict,
                    **hybrid,
                    **strict_allowed,
                }
            )
            self._emit_progress(
                stage="eval_drift.windows",
                status="update",
                message="Evaluating drift windows",
                current=len(rows),
                total=len(windows),
                payload={"window_count": len(window), "draws": self.mc_draws},
            )
        self._emit_progress(
            stage="eval_drift.windows",
            status="done",
            message="Drift evaluation completed",
            current=len(rows),
            total=len(windows),
            payload={"windows": len(rows), "draws": self.mc_draws},
        )
        return rows

    def _resolve_drift_step(self) -> int:
        return self.drift_window_sliding or self.drift_window_size

    @staticmethod
    def _ordered_trace_indices(records: Sequence[UniformMaskEvaluationRecord]) -> list[int]:
        first_seen: dict[int, tuple[float, int]] = {}
        for order, record in enumerate(records):
            current = first_seen.get(record.trace_idx)
            candidate = (record.trace_start_ts, order)
            if current is None or candidate < current:
                first_seen[record.trace_idx] = candidate
        return [
            trace_idx
            for trace_idx, _ in sorted(first_seen.items(), key=lambda item: (item[1][0], item[1][1], item[0]))
        ]

    def _emit_progress(
        self,
        *,
        stage: str,
        status: str,
        message: str,
        current: int | float | None = None,
        total: int | float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            stage=stage,
            status=status,
            message=message,
            current=current,
            total=total,
            payload=payload or {},
        )

    @staticmethod
    def _summary(prefix: str, values: Sequence[float]) -> dict[str, float]:
        if not values:
            return {
                f"{prefix}_mc_mean": 0.0,
                f"{prefix}_mc_std": 0.0,
                f"{prefix}_mc_sampling_uncertainty_95_low": 0.0,
                f"{prefix}_mc_sampling_uncertainty_95_high": 0.0,
            }
        avg = mean(values)
        if len(values) > 1:
            variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        half_width = 1.96 * std / math.sqrt(len(values))
        return {
            f"{prefix}_mc_mean": avg,
            f"{prefix}_mc_std": std,
            f"{prefix}_mc_sampling_uncertainty_95_low": max(0.0, avg - half_width),
            f"{prefix}_mc_sampling_uncertainty_95_high": min(1.0, avg + half_width),
        }

    @staticmethod
    def _macro_f1(true_keys: Sequence[str], pred_keys: Sequence[str]) -> float:
        labels = sorted(set(true_keys).union(pred_keys))
        if not labels:
            return 0.0
        true_counts = Counter(true_keys)
        pred_counts = Counter(pred_keys)
        true_pred_counts = Counter(zip(true_keys, pred_keys))
        scores = []
        for label in labels:
            tp = true_pred_counts[(label, label)]
            fp = pred_counts[label] - tp
            fn = true_counts[label] - tp
            if tp == 0 and fp == 0 and fn == 0:
                continue
            scores.append((2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) else 0.0)
        return mean(scores) if scores else 0.0

    @staticmethod
    def _scalar_int(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.detach().cpu().view(-1)[0].item()) if value.numel() else 0
        return int(value)

    @staticmethod
    def _scalar_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().view(-1)[0].item()) if value.numel() else 0.0
        return float(value)
