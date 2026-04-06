"""Assign process version labels to XES traces/events using config-driven policy.

Example:
  python main.py add-version2xes --config configs/tools/add_version2xes_re2.5.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lxml import etree
import yaml


_XES_ATTR_TAGS = {"string", "date", "int", "float", "boolean", "id", "list"}


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _qualify_like(parent: etree._Element, local_name: str) -> str:
    tag = parent.tag if isinstance(parent.tag, str) else ""
    if tag.startswith("{") and "}" in tag:
        ns = tag[1 : tag.index("}")]
        return f"{{{ns}}}{local_name}"
    return local_name


def _upsert_xes_attribute(parent: etree._Element, key: str, value: str) -> bool:
    """Set XES attribute on parent; return True if existing key was overwritten."""
    for child in parent:
        if _local_name(child.tag) not in _XES_ATTR_TAGS:
            continue
        if str(child.get("key", "")).strip() != key:
            continue
        child.set("value", value)
        return True
    attr = etree.Element(_qualify_like(parent, "string"))
    attr.set("key", key)
    attr.set("value", value)
    parent.insert(0, attr)
    return False


def _collect_traces(root: etree._Element) -> List[etree._Element]:
    traces: List[etree._Element] = []
    for elem in root.iter():
        if _local_name(elem.tag) == "trace":
            traces.append(elem)
    return traces


def _normalize_scope(value: Any) -> str:
    scope = str(value).strip().lower() or "both"
    if scope not in {"trace", "event", "both"}:
        raise ValueError("scope must be one of {'trace','event','both'}.")
    return scope


def _resolve_chunk_size(total_traces: int, policy: Dict[str, Any]) -> int:
    raw_chunk_size = policy.get("chunk_size")
    if raw_chunk_size not in (None, ""):
        chunk_size = int(raw_chunk_size)
        if chunk_size <= 0:
            raise ValueError("policy.chunk_size must be > 0.")
        return chunk_size

    raw_chunk_percent = policy.get("chunk_percent", 0.1)
    chunk_percent = float(raw_chunk_percent)
    if chunk_percent <= 0.0 or chunk_percent > 1.0:
        raise ValueError("policy.chunk_percent must be within (0.0, 1.0].")

    chunk_size = int(total_traces * chunk_percent)
    return max(1, chunk_size)


def _resolve_policy(policy: Dict[str, Any], total_traces: int) -> Tuple[List[str], int]:
    policy_type = str(policy.get("type", "alternating_chunks")).strip().lower() or "alternating_chunks"
    if policy_type != "alternating_chunks":
        raise ValueError("Only policy.type='alternating_chunks' is currently supported.")

    versions_raw = policy.get("versions", ["v1", "v2"])
    if not isinstance(versions_raw, list) or not versions_raw:
        raise ValueError("policy.versions must be a non-empty list.")
    versions = [str(item).strip() for item in versions_raw if str(item).strip()]
    if len(versions) < 2:
        raise ValueError("policy.versions must contain at least two non-empty values (e.g., ['v1','v2']).")

    chunk_size = _resolve_chunk_size(total_traces=total_traces, policy=policy)
    return versions, chunk_size


def _read_config(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping.")
    return payload


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    input_path = Path(str(cfg.get("input", "")).strip()).expanduser().resolve()
    output_path = Path(str(cfg.get("output", "")).strip()).expanduser().resolve()
    if not str(input_path):
        raise ValueError("Missing required config key: input")
    if not str(output_path):
        raise ValueError("Missing required config key: output")
    if not input_path.exists():
        raise FileNotFoundError(f"Input XES not found: {input_path}")

    version_key = str(cfg.get("version_key", "concept:version")).strip() or "concept:version"
    scope = _normalize_scope(cfg.get("scope", "both"))
    dry_run = bool(cfg.get("dry_run", False))

    policy_raw = cfg.get("policy", {})
    policy = dict(policy_raw) if isinstance(policy_raw, dict) else {}

    parser = etree.XMLParser(remove_comments=False, recover=False, huge_tree=True)
    tree = etree.parse(str(input_path), parser=parser)
    root = tree.getroot()
    traces = _collect_traces(root)
    total_traces = len(traces)
    if total_traces == 0:
        raise ValueError("No <trace> elements found in input XES.")

    versions, chunk_size = _resolve_policy(policy=policy, total_traces=total_traces)

    trace_updates = 0
    event_updates = 0
    version_counts: Dict[str, int] = {version: 0 for version in versions}

    for trace_idx, trace in enumerate(traces):
        version = versions[(trace_idx // chunk_size) % len(versions)]
        version_counts[version] = version_counts.get(version, 0) + 1

        if scope in {"trace", "both"}:
            _upsert_xes_attribute(trace, version_key, version)
            trace_updates += 1

        if scope in {"event", "both"}:
            for child in trace:
                if _local_name(child.tag) != "event":
                    continue
                _upsert_xes_attribute(child, version_key, version)
                event_updates += 1

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(str(output_path), xml_declaration=True, encoding="utf-8", pretty_print=True)

    return {
        "status": "ok",
        "mode": "add-version2xes",
        "input": str(input_path),
        "output": str(output_path),
        "dry_run": dry_run,
        "version_key": version_key,
        "scope": scope,
        "total_traces": total_traces,
        "policy": {
            "type": "alternating_chunks",
            "versions": versions,
            "chunk_size": chunk_size,
            "chunk_percent": float(policy.get("chunk_percent", 0.0) or 0.0),
        },
        "trace_updates": trace_updates,
        "event_updates": event_updates,
        "assigned_trace_counts": version_counts,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign version labels to XES traces/events from config.")
    parser.add_argument("--config", required=True, help="Path to YAML config for add-version2xes.")
    parser.add_argument("--out", default="", help="Optional path for JSON summary output.")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg_path = Path(str(args.config)).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = _read_config(cfg_path)
    summary = run(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_text = str(args.out).strip()
    if out_text:
        out_path = Path(out_text).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
