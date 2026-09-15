#!/usr/bin/env python3
"""Reproduce EV-01 summaries from frozen article metric exports.

This script is copied into the article evidence directory before execution.  It
only reads source artifacts and writes derived audit tables next to itself.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


R1 = Path(__file__).resolve().parents[2]
ROOT = R1 / "Export_metrics"
CDLG_TOOL = Path(r"C:\Users\korsr\PycharmProjects\cdlg_tool-main")
REGISTRY = CDLG_TOOL / "outputs/campaigns/cdlg-structural-v1/campaign_registry.csv"
OUT = Path(__file__).resolve().parent
METRICS = ("drift_window_strict_macro_f1", "drift_window_test_oos",
           "drift_window_start_ts", "drift_window_end_ts")
WINDOW_SIZE = 100
WINDOW_STEP = 10
LOAN_XES = Path(r"C:\Users\korsr\PycharmProjects\bpm_prediction\outputs\simulation\loan_v1_v5_complex_simulated.xes")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def sign_flip(values: list[float]) -> float:
    """Two-sided exact sign-flip p value for absolute mean difference."""
    observed = abs(mean(values))
    total = 2 ** len(values)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        if abs(sum(v * sign for v, sign in zip(values, signs)) / len(values)) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def holm(records: list[dict]) -> None:
    ordered = sorted(enumerate(records), key=lambda item: item[1]["p_raw"])
    running = 0.0
    count = len(ordered)
    for rank, (_, record) in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * record["p_raw"])
        running = max(running, adjusted)
        record["p_holm"] = running


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["paper_model"], row["run_id"], row["step"]


def load_series(benchmark: str) -> tuple[dict, list[dict]]:
    folder = ROOT / "article_run_metrics" / benchmark / "drift"
    series: dict[str, dict[tuple[str, str, str], dict[str, str]]] = {}
    for metric in METRICS:
        series[metric] = {key(row): row for row in read_csv(folder / f"{metric}.csv")}
    manifest = read_csv(folder / "run_manifest.csv")
    return series, manifest


def cdlg_registry() -> dict[str, dict[str, str]]:
    rows = read_csv(REGISTRY)
    return {row["logical_id"]: row for row in rows if row["status"] == "published"}


def cdlg_id(run_name: str) -> str:
    match = re.search(r"_CDLG-(simple|medium|complex)([1-5])_", run_name, re.I)
    if not match:
        raise ValueError(f"Cannot resolve CDLG process from {run_name}")
    return f"cdlg-{match.group(1).lower()}{match.group(2)}"


def boundaries_from_bundle(row: dict[str, str]) -> list[tuple[str, int, int]]:
    report = CDLG_TOOL / row["bundle_path"] / "reports/drift_metrics.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    return [(item["version"], int(item["start_index"]), int(item["end_index"]))
            for item in payload["boundaries"]]


def classify_window(start: int, end: int, bounds: list[tuple[str, int, int]]) -> str | None:
    for version, low, high in bounds:
        if low <= start <= end <= high:
            return version
    return None


def loan_trace_versions() -> list[str]:
    """Read trace-level version labels from the frozen loan simulation log."""
    result: list[tuple[float, str]] = []
    for _, element in ET.iterparse(LOAN_XES, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "trace":
            continue
        version = None
        start = None
        for child in element:
            tag = child.tag.rsplit("}", 1)[-1]
            if child.attrib.get("key") == "concept:version" and tag == "string":
                version = child.attrib.get("value")
            if child.attrib.get("key") == "sim:case_start_time" and tag == "date":
                start = child.attrib.get("value")
        if version is None or start is None:
            raise ValueError("Loan trace lacks a trace-level version or case-start timestamp")
        result.append((datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp(), version))
        element.clear()
    return [version for _, version in sorted(result)]


def run_rows(benchmark: str, series: dict, manifest: list[dict], registry: dict, loan_versions: list[str]) -> list[dict]:
    by_run = {row["run_id"]: row for row in manifest}
    all_keys = set.intersection(*(set(data) for data in series.values()))
    rows: list[dict] = []
    for model, run_id, step in sorted(all_keys, key=lambda item: (item[1], int(item[2]))):
        run = by_run[run_id]
        name = run["run_name"]
        if benchmark == "CDLG":
            entity = cdlg_id(name)
            bounds = boundaries_from_bundle(registry[entity])
            complexity = registry[entity]["cdlg_complexity"]
        else:
            entity = f"loan-seed-{run['seed']}"
            complexity = "loan"
            bounds = []
        # The 100-trace/10-trace-step schedule is recorded in the saved preset;
        # MOU's explicit index exports independently validate it for CDLG.
        start = int(step) * WINDOW_STEP
        end = start + WINDOW_SIZE - 1
        if benchmark == "CDLG":
            version = classify_window(start, end, bounds)
        else:
            version = loan_versions[start] if end < len(loan_versions) and len(set(loan_versions[start:end + 1])) == 1 else None
        rows.append({
            "benchmark": benchmark, "entity_id": entity, "complexity": complexity,
            "paper_model": model, "seed": run["seed"], "run_id": run_id,
            "step": int(step), "window_start_trace_idx": start,
            "window_end_trace_idx": end, "version": version or "mixed_or_unverified",
            "strict_macro_f1": float(series["drift_window_strict_macro_f1"][(model, run_id, step)]["value"]),
            "oos": float(series["drift_window_test_oos"][(model, run_id, step)]["value"]),
        })
    return rows


def scope_summaries(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark"], row["entity_id"], row["complexity"], row["paper_model"])].append(row)
    entity_scopes: list[dict] = []
    for identity, values in grouped.items():
        benchmark, entity, complexity, model = identity
        by_version: dict[str, list[dict]] = defaultdict(list)
        for row in values:
            by_version[row["version"]].append(row)
        scopes = {"full_stream": values}
        eligible = {version: by_version[version] for version in ("v1", "v2", "v3", "v4", "v5")}
        if all(eligible.values()):
            for label, versions in (("all_versions", ("v1", "v2", "v3", "v4", "v5")),
                                    ("future_versions", ("v3", "v4", "v5"))):
                # First average windows within each version, then versions equally.
                scopes[label] = [
                    {"strict_macro_f1": mean([item["strict_macro_f1"] for item in eligible[v]]),
                     "oos": mean([item["oos"] for item in eligible[v]])}
                    for v in versions
                ]
        for scope, items in scopes.items():
            entity_scopes.append({
                "benchmark": benchmark, "entity_id": entity, "complexity": complexity,
                "paper_model": model, "scope": scope, "window_or_version_units": len(items),
                "strict_macro_f1": mean([item["strict_macro_f1"] for item in items]),
                "oos": mean([item["oos"] for item in items]),
            })
    summary: list[dict] = []
    grouped_summary: dict[tuple, list[dict]] = defaultdict(list)
    for row in entity_scopes:
        grouped_summary[(row["benchmark"], row["complexity"], row["paper_model"], row["scope"])].append(row)
    for identity, values in sorted(grouped_summary.items()):
        benchmark, complexity, model, scope = identity
        for metric in ("strict_macro_f1", "oos"):
            vals = [row[metric] for row in values]
            summary.append({"benchmark": benchmark, "complexity": complexity, "paper_model": model,
                            "scope": scope, "metric": metric, "n_entities": len(vals),
                            "mean": mean(vals), "sample_sd": sample_sd(vals),
                            "uncertainty": "between-process sample SD" if benchmark == "CDLG" else "between-seed sample SD"})
    # CDLG overall is a summary of its three strata, not an additional sample.
    for model in sorted({row["paper_model"] for row in entity_scopes if row["benchmark"] == "CDLG"}):
        for scope in sorted({row["scope"] for row in entity_scopes if row["benchmark"] == "CDLG"}):
            values = [row for row in entity_scopes if row["benchmark"] == "CDLG" and row["paper_model"] == model and row["scope"] == scope]
            for metric in ("strict_macro_f1", "oos"):
                vals = [row[metric] for row in values]
                summary.append({"benchmark": "CDLG", "complexity": "overall_15_processes", "paper_model": model,
                                "scope": scope, "metric": metric, "n_entities": len(vals),
                                "mean": mean(vals), "sample_sd": sample_sd(vals),
                                "uncertainty": "between-process sample SD"})
    return entity_scopes, summary


def significance(entity_scopes: list[dict]) -> list[dict]:
    lookup = {(r["benchmark"], r["entity_id"], r["scope"], r["paper_model"]): r for r in entity_scopes}
    output: list[dict] = []
    for benchmark in ("CDLG", "loan"):
        for scope in ("future_versions", "all_versions"):
            family: list[dict] = []
            entities = sorted({r["entity_id"] for r in entity_scopes if r["benchmark"] == benchmark and r["scope"] == scope})
            for control in ("GATv2+Mask", "EOPKG-WI"):
                for metric in ("strict_macro_f1", "oos"):
                    paired: list[float] = []
                    for entity in entities:
                        eopkg = lookup[(benchmark, entity, scope, "EOPKG")][metric]
                        baseline = lookup[(benchmark, entity, scope, control)][metric]
                        paired.append(eopkg - baseline if metric == "strict_macro_f1" else baseline - eopkg)
                    sd = sample_sd(paired)
                    family.append({"benchmark": benchmark, "scope": scope, "metric": metric,
                                   "contrast": f"EOPKG vs {control}", "n": len(paired),
                                   "mean_difference": mean(paired), "sd_difference": sd,
                                   "dz": mean(paired) / sd if sd and not math.isnan(sd) and sd != 0 else "NA",
                                   "wins": sum(x > 0 for x in paired), "ties": sum(x == 0 for x in paired),
                                   "losses": sum(x < 0 for x in paired), "p_raw": sign_flip(paired)})
            holm(family)
            output.extend(family)
    return output


def source_inventory(registry: dict) -> list[dict]:
    records = []
    for benchmark in ("CDLG", "loan"):
        folder = ROOT / "article_run_metrics" / benchmark / "drift"
        for name in ("run_manifest.csv", *[f"{m}.csv" for m in METRICS]):
            path = folder / name
            records.append({"role": f"{benchmark} metric export", "path": str(path), "sha256": sha256(path)})
    records.append({"role": "CDLG generation registry", "path": str(REGISTRY), "sha256": sha256(REGISTRY)})
    for logical_id, record in sorted(registry.items()):
        report = CDLG_TOOL / record["bundle_path"] / "reports/drift_metrics.json"
        records.append({"role": f"CDLG drift metadata ({logical_id})", "path": str(report), "sha256": sha256(report)})
    return records


def dataset_summary(registry: dict) -> list[dict]:
    rows = []
    for logical_id, record in sorted(registry.items()):
        bundle = CDLG_TOOL / record["bundle_path"]
        drift = json.loads((bundle / "reports/drift_metrics.json").read_text(encoding="utf-8"))
        alignment = json.loads((bundle / "reports/topology_alignment.json").read_text(encoding="utf-8"))
        task_counts = [len(item["bpmn_activities"]) for item in alignment["versions"].values()]
        edge_counts = []
        for bpmn in (bundle / "models/bpmn").glob("*.bpmn"):
            root = ET.parse(bpmn).getroot()
            edge_counts.append(sum(element.tag.rsplit("}", 1)[-1] == "sequenceFlow" for element in root.iter()))
        rows.append({"logical_id": logical_id, "complexity": record["cdlg_complexity"],
                     "process_key": record["process_key"], "structural_hash": record["structural_hash"],
                     "trace_count": drift["published_trace_count"], "versions": len(drift["boundaries"]),
                     "traces_per_version": "/".join(str(drift["version_counts"][f"v{i}"]) for i in range(1, 6)),
                     "drift_count": drift["drift_count"], "drift_type": drift["drift_type"],
                     "task_count_min": min(task_counts), "task_count_max": max(task_counts),
                     "sequence_flow_min": min(edge_counts), "sequence_flow_max": max(edge_counts),
                     "unique_process_tree_hashes": len({item["process_tree_sha256"] for item in drift["snapshots"]})})
    return rows


def main() -> None:
    registry = cdlg_registry()
    if len(registry) != 15:
        raise ValueError(f"Expected 15 published CDLG processes, found {len(registry)}")
    all_rows: list[dict] = []
    loan_versions = loan_trace_versions()
    for benchmark in ("CDLG", "loan"):
        series, manifest = load_series(benchmark)
        all_rows.extend(run_rows(benchmark, series, manifest, registry, loan_versions))
    fields = ["benchmark", "entity_id", "complexity", "paper_model", "seed", "run_id", "step",
              "window_start_trace_idx", "window_end_trace_idx", "version", "strict_macro_f1", "oos"]
    write_csv(OUT / "version_metrics.csv", all_rows, fields)
    entity_scopes, summary = scope_summaries(all_rows)
    write_csv(OUT / "entity_scope_metrics.csv", entity_scopes, list(entity_scopes[0]))
    write_csv(OUT / "scope_summary.csv", summary, list(summary[0]))
    tests = significance(entity_scopes)
    write_csv(OUT / "paired_significance.csv", tests, list(tests[0]))
    inventory = source_inventory(registry)
    write_csv(OUT / "source_inventory.csv", inventory, ["role", "path", "sha256"])
    datasets = dataset_summary(registry)
    write_csv(OUT / "dataset_summary.csv", datasets, list(datasets[0]))
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)),
                "cdlg_processes": len(registry), "loan_trace_count": len(loan_versions),
                "window_schedule": {"size": WINDOW_SIZE, "step": WINDOW_STEP},
                "note": "Version membership uses only windows whose 100 trace-level labels agree; boundaries are not inferred from figures."}
    (OUT / "analysis_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
