"""Offline audit for topology drift, new nodes, and fixed-head failure modes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.application.services.topology_drift_audit import (
    PredictionAttribution,
    PredictionRecord,
    VersionTopologyDiff,
    VersionTopologySnapshot,
    attribute_prediction_error,
    diff_topology_versions,
    parse_bpmn_topology_snapshot,
    parse_xes_prefix_last_activity_lookup,
)


DEFAULT_BPMN_FILES = {
    "v1": "loan_v1.bpmn",
    "v2": "loan_v2_re.bpmn",
    "v3": "loan_v3_re_pl.bpmn",
    "v4": "loan_v4_re_pl_cb.bpmn",
}


def main() -> int:
    args = _parse_args()
    mlruns_dir = Path(args.mlruns_dir)
    experiment_dir = mlruns_dir / str(args.experiment_id)
    output_root = Path(args.output_dir)
    audit_id = args.audit_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_topology_drift"
    output_dir = output_root / audit_id
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = _load_bpmn_snapshots(Path(args.bpmn_dir), label_attr=args.label_attr)
    train_versions = _resolve_train_versions(
        train_run_dir=experiment_dir / args.train_run,
        explicit=args.train_versions,
        available_versions=list(snapshots),
    )
    eval_versions = _resolve_eval_versions(args.eval_versions, available_versions=list(snapshots))
    train_activity_counts = _train_activity_counts(snapshots, train_versions)
    topology_diffs = _build_topology_diffs(snapshots, train_versions=train_versions, eval_versions=eval_versions)
    prefix_last_activity_lookup = _load_prefix_last_activity_lookup(
        log_path=_resolve_log_path(args.log_path, dataset=args.dataset),
    )

    baseline_trace_rows = _load_trace_records(
        experiment_dir=experiment_dir,
        run_id=args.baseline_run,
        model_label="baseline",
        prefix_last_activity_lookup=prefix_last_activity_lookup,
    )
    structural_trace_rows = _load_trace_records(
        experiment_dir=experiment_dir,
        run_id=args.structural_run,
        model_label="structural",
        prefix_last_activity_lookup=prefix_last_activity_lookup,
    )
    attributions = _attribute_records(
        baseline_trace_rows + structural_trace_rows,
        topology_diffs=topology_diffs,
        train_activity_counts=train_activity_counts,
    )

    version_topology_rows = _version_topology_rows(snapshots, eval_versions=eval_versions)
    activity_diff_rows = _activity_diff_rows(topology_diffs)
    transition_diff_rows = _transition_diff_rows(topology_diffs)
    attribution_rows = [_attribution_to_row(item) for item in attributions]
    by_version_rows = _by_version_rows(attributions)
    summary = _summary_payload(
        audit_id=audit_id,
        dataset=args.dataset,
        train_versions=train_versions,
        eval_versions=eval_versions,
        baseline_run=args.baseline_run,
        structural_run=args.structural_run,
        topology_diffs=topology_diffs,
        attributions=attributions,
        prediction_error_attribution_available=bool(attributions),
    )

    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "version_topology_diff.csv", version_topology_rows)
    _write_csv(output_dir / "activity_label_diff.csv", activity_diff_rows)
    _write_csv(output_dir / "transition_diff.csv", transition_diff_rows)
    _write_csv(output_dir / "prediction_error_attribution.csv", attribution_rows)
    _write_csv(output_dir / "by_version_metrics.csv", by_version_rows)
    (output_dir / "report.md").write_text(
        _render_report(summary, activity_diff_rows, transition_diff_rows, by_version_rows),
        encoding="utf-8",
    )
    print(f"Topology drift audit written to {output_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlruns-dir", default="mlruns")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--structural-run", required=True)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--output-dir", default="outputs/audits/topology_drift")
    parser.add_argument("--audit-id", default="")
    parser.add_argument("--dataset", default="loan_v1_v4_simulated")
    parser.add_argument("--bpmn-dir", default="data/camunda_exports/bpmn_xml")
    parser.add_argument("--label-attr", choices=["id", "name"], default="id")
    parser.add_argument("--log-path", default="")
    parser.add_argument("--train-versions", default="")
    parser.add_argument("--eval-versions", default="")
    return parser.parse_args()


def _load_bpmn_snapshots(bpmn_dir: Path, *, label_attr: str) -> Dict[str, VersionTopologySnapshot]:
    snapshots: Dict[str, VersionTopologySnapshot] = {}
    for version, filename in DEFAULT_BPMN_FILES.items():
        path = bpmn_dir / filename
        if path.exists():
            snapshots[version] = parse_bpmn_topology_snapshot(path, version=version, label_attr=label_attr)
    if not snapshots:
        raise FileNotFoundError(f"No BPMN files found in {bpmn_dir}.")
    return snapshots


def _resolve_train_versions(
    *,
    train_run_dir: Path,
    explicit: str,
    available_versions: Sequence[str],
) -> List[str]:
    if explicit.strip():
        return _split_versions(explicit)
    raw = _read_run_param(train_run_dir, "data_train_versions") or ""
    versions: List[str] = []
    for item in raw.split(","):
        token = item.strip()
        if not token or token == "none":
            continue
        versions.append(token.split(":", 1)[0])
    return versions or [str(available_versions[0])]


def _resolve_eval_versions(raw: str, *, available_versions: Sequence[str]) -> List[str]:
    return _split_versions(raw) if raw.strip() else [str(item) for item in available_versions]


def _split_versions(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _read_run_param(run_dir: Path, key: str) -> str | None:
    path = run_dir / "params" / key
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _train_activity_counts(
    snapshots: Mapping[str, VersionTopologySnapshot],
    train_versions: Sequence[str],
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for version in train_versions:
        snapshot = snapshots.get(str(version))
        if snapshot is None:
            continue
        for activity in snapshot.activities:
            counts[str(activity)] += 1
    return dict(counts)


def _build_topology_diffs(
    snapshots: Mapping[str, VersionTopologySnapshot],
    *,
    train_versions: Sequence[str],
    eval_versions: Sequence[str],
) -> Dict[str, VersionTopologyDiff]:
    result: Dict[str, VersionTopologyDiff] = {}
    ordered = list(snapshots)
    train_set = {str(item) for item in train_versions}
    fallback_source = str(train_versions[-1]) if train_versions else ordered[0]
    for target_version in eval_versions:
        source_version = _reference_version_for_target(
            target_version=str(target_version),
            ordered_versions=ordered,
            train_versions=train_set,
            fallback=fallback_source,
        )
        source = snapshots.get(source_version)
        target = snapshots.get(str(target_version))
        if source is None or target is None:
            continue
        result[str(target_version)] = diff_topology_versions(source, target)
    return result


def _reference_version_for_target(
    *,
    target_version: str,
    ordered_versions: Sequence[str],
    train_versions: set[str],
    fallback: str,
) -> str:
    if target_version in train_versions:
        return target_version
    try:
        target_index = list(ordered_versions).index(target_version)
    except ValueError:
        return fallback
    candidates = [version for version in ordered_versions[:target_index] if version in train_versions]
    return candidates[-1] if candidates else fallback


def _resolve_log_path(raw: str, *, dataset: str) -> Path | None:
    if str(raw).strip():
        return Path(raw)
    candidate = Path("outputs") / "simulation" / f"{dataset}.xes"
    return candidate if candidate.exists() else None


def _load_prefix_last_activity_lookup(*, log_path: Path | None) -> Dict[tuple[int, int], str]:
    if log_path is None or not log_path.exists():
        return {}
    return parse_xes_prefix_last_activity_lookup(log_path)


def _load_trace_records(
    *,
    experiment_dir: Path,
    run_id: str,
    model_label: str,
    prefix_last_activity_lookup: Mapping[tuple[int, int], str] | None = None,
) -> List[PredictionRecord]:
    artifacts_dir = experiment_dir / run_id / "artifacts"
    paths = sorted(artifacts_dir.glob(f"structural_traces_run_{run_id}_*.jsonl"))
    records: List[PredictionRecord] = []
    prefix_lookup = dict(prefix_last_activity_lookup or {})
    for path in paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            payload = json.loads(line)
            attrs = payload.get("attributes", {}) if isinstance(payload, dict) else {}
            inputs = payload.get("inputs", {}) if isinstance(payload, dict) else {}
            outputs = payload.get("outputs", {}) if isinstance(payload, dict) else {}
            prediction = outputs.get("prediction", {}) if isinstance(outputs, dict) else {}
            sample = inputs.get("sample", {}) if isinstance(inputs, dict) else {}
            mask = inputs.get("mask", {}) if isinstance(inputs, dict) else {}
            target_label = str(prediction.get("target_label", attrs.get("target_index", "")))
            pred_label = str(prediction.get("pred_label", attrs.get("pred_index", "")))
            trace_idx = int(sample.get("trace_idx", line_no))
            prefix_len_raw = sample.get("prefix_len")
            prefix_len = int(prefix_len_raw) if prefix_len_raw is not None else -1
            prefix_last_activity = str(sample.get("prefix_last_activity", "")).strip()
            if not prefix_last_activity or prefix_last_activity == "__unknown__":
                prefix_last_activity = str(
                    prefix_lookup.get((trace_idx, prefix_len), "__unknown__")
                )
            records.append(
                PredictionRecord(
                    run_id=run_id,
                    model_label=model_label,
                    trace_idx=trace_idx,
                    step=int(sample.get("global_index", line_no)),
                    process_version=str(attrs.get("process_version", sample.get("process_version", ""))),
                    prefix_last_activity=prefix_last_activity,
                    target_label=target_label,
                    pred_label=pred_label,
                    strict_correct=bool(attrs.get("strict_correct", False)),
                    pred_in_mask=bool(attrs.get("pred_in_mask", attrs.get("prediction_in_mask", False))),
                    target_in_mask=bool(attrs.get("target_in_mask", mask.get("target_in_mask", False))),
                    strict_error_but_allowed=bool(attrs.get("strict_error_but_allowed", False)),
                    mask_cardinality=int(float(attrs.get("mask_cardinality", mask.get("mask_cardinality", 0)))),
                )
            )
    return records


def _attribute_records(
    records: Iterable[PredictionRecord],
    *,
    topology_diffs: Mapping[str, VersionTopologyDiff],
    train_activity_counts: Mapping[str, int],
) -> List[PredictionAttribution]:
    attributions: List[PredictionAttribution] = []
    for record in records:
        diff = topology_diffs.get(record.process_version)
        if diff is None:
            continue
        attributions.append(
            attribute_prediction_error(
                record,
                train_activity_counts=train_activity_counts,
                topology_diff=diff,
            )
        )
    return attributions


def _version_topology_rows(
    snapshots: Mapping[str, VersionTopologySnapshot],
    *,
    eval_versions: Sequence[str],
) -> List[Dict[str, Any]]:
    rows = []
    for version in eval_versions:
        snapshot = snapshots.get(str(version))
        if snapshot is None:
            continue
        rows.append(
            {
                "version": version,
                "activities_total": len(snapshot.activities),
                "transitions_total": len(snapshot.transitions),
            }
        )
    return rows


def _activity_diff_rows(diffs: Mapping[str, VersionTopologyDiff]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for version, diff in diffs.items():
        for activity in sorted(diff.added_activities):
            rows.append({"version": version, "change": "added", "activity": activity})
        for activity in sorted(diff.removed_activities):
            rows.append({"version": version, "change": "removed", "activity": activity})
    return rows


def _transition_diff_rows(diffs: Mapping[str, VersionTopologyDiff]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for version, diff in diffs.items():
        for src, dst in sorted(diff.added_transitions):
            rows.append({"version": version, "change": "added", "source": src, "target": dst})
        for src, dst in sorted(diff.removed_transitions):
            rows.append({"version": version, "change": "removed", "source": src, "target": dst})
    return rows


def _attribution_to_row(item: PredictionAttribution) -> Dict[str, Any]:
    record = item.record
    return {
        "run_id": record.run_id,
        "model_label": record.model_label,
        "trace_idx": record.trace_idx,
        "step": record.step,
        "process_version": record.process_version,
        "prefix_last_activity": record.prefix_last_activity,
        "target_label": record.target_label,
        "pred_label": record.pred_label,
        "strict_correct": record.strict_correct,
        "pred_in_mask": record.pred_in_mask,
        "target_in_mask": record.target_in_mask,
        "strict_error_but_allowed": record.strict_error_but_allowed,
        "is_oos": not record.pred_in_mask,
        "mask_cardinality": record.mask_cardinality,
        "target_seen_in_train": item.target_seen_in_train,
        "target_count_in_train": item.target_count_in_train,
        "target_is_new_in_eval_version": item.target_is_new_in_eval_version,
        "pred_removed_in_eval_version": item.pred_removed_in_eval_version,
        "prefix_transition_changed": item.prefix_transition_changed,
        "old_allowed_targets": "|".join(sorted(item.old_allowed_targets)),
        "new_allowed_targets": "|".join(sorted(item.new_allowed_targets)),
        "error_bucket": item.error_bucket,
    }


def _by_version_rows(attributions: Sequence[PredictionAttribution]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], list[PredictionAttribution]] = defaultdict(list)
    for item in attributions:
        grouped[(item.record.model_label, item.record.process_version)].append(item)
    rows: List[Dict[str, Any]] = []
    for (model_label, version), items in sorted(grouped.items()):
        total = len(items)
        bucket_counts = Counter(item.error_bucket for item in items)
        rows.append(
            {
                "model_label": model_label,
                "process_version": version,
                "samples": total,
                "strict_correct_rate": _rate(sum(1 for item in items if item.record.strict_correct), total),
                "oos_rate": _rate(sum(1 for item in items if not item.record.pred_in_mask), total),
                "unseen_target_rate": _rate(sum(1 for item in items if item.error_bucket == "unseen_target_class"), total),
                "removed_prediction_rate": _rate(
                    sum(1 for item in items if item.error_bucket == "removed_prediction_class"),
                    total,
                ),
                "changed_transition_rate": _rate(
                    sum(1 for item in items if item.error_bucket == "changed_transition_zone"),
                    total,
                ),
                "bucket_counts": json.dumps(dict(bucket_counts), ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def _summary_payload(
    *,
    audit_id: str,
    dataset: str,
    train_versions: Sequence[str],
    eval_versions: Sequence[str],
    baseline_run: str,
    structural_run: str,
    topology_diffs: Mapping[str, VersionTopologyDiff],
    attributions: Sequence[PredictionAttribution],
    prediction_error_attribution_available: bool,
) -> Dict[str, Any]:
    added_activities = sorted({activity for diff in topology_diffs.values() for activity in diff.added_activities})
    removed_activities = sorted({activity for diff in topology_diffs.values() for activity in diff.removed_activities})
    bucket_counts = Counter(item.error_bucket for item in attributions)
    fixed_head_limitation = bool(bucket_counts.get("unseen_target_class", 0) or added_activities)
    changed_transition_failures = int(bucket_counts.get("changed_transition_zone", 0))
    return {
        "audit_id": audit_id,
        "dataset": dataset,
        "train_versions": list(train_versions),
        "eval_versions": list(eval_versions),
        "runs": {"baseline": baseline_run, "structural": structural_run},
        "prediction_error_attribution_available": prediction_error_attribution_available,
        "findings": {
            "new_activity_labels_present": bool(added_activities),
            "removed_activity_labels_present": bool(removed_activities),
            "added_activity_labels": added_activities,
            "removed_activity_labels": removed_activities,
            "changed_transition_error_count": changed_transition_failures,
            "unseen_target_error_count": int(bucket_counts.get("unseen_target_class", 0)),
            "fixed_head_limitation_detected": fixed_head_limitation,
            "error_buckets": dict(bucket_counts),
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _render_report(
    summary: Mapping[str, Any],
    activity_rows: Sequence[Mapping[str, Any]],
    transition_rows: Sequence[Mapping[str, Any]],
    by_version_rows: Sequence[Mapping[str, Any]],
) -> str:
    findings = summary.get("findings", {})
    return "\n".join(
        [
            "# Topology Drift Audit Report",
            "",
            "## Scope",
            "",
            "Offline audit for added/removed topology candidates, changed transitions, and fixed-head risk.",
            "",
            "## Runs",
            "",
            f"- baseline: `{summary.get('runs', {}).get('baseline')}`",
            f"- structural: `{summary.get('runs', {}).get('structural')}`",
            "",
            "## Version Topology Diff",
            "",
            f"- new_activity_labels_present: `{findings.get('new_activity_labels_present')}`",
            f"- removed_activity_labels_present: `{findings.get('removed_activity_labels_present')}`",
            f"- activity_diff_rows: `{len(activity_rows)}`",
            f"- transition_diff_rows: `{len(transition_rows)}`",
            "",
            "## Prediction Error Attribution",
            "",
            f"- prediction_error_attribution_available: `{summary.get('prediction_error_attribution_available')}`",
            f"- error_buckets: `{json.dumps(findings.get('error_buckets', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- by_version_rows: `{len(by_version_rows)}`",
            "",
            "## Fixed Classifier Head Risk",
            "",
            f"- fixed_head_limitation_detected: `{findings.get('fixed_head_limitation_detected')}`",
            "",
            "## Methodology Implications",
            "",
            "If added activity labels or unseen target errors are present, fixed parametric classification is not enough for business-valid zero-shot drift.",
            "",
            "## Recommended Next Experiments",
            "",
            "1. Compare errors in `prediction_error_attribution.csv` against changed successor zones.",
            "2. If unseen targets dominate, plan dynamic topology-conditioned candidate scoring.",
            "3. If masks dominate, fix topology projection and mask alignment first.",
            "",
        ]
    )


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
