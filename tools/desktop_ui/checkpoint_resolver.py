from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT_VALUE = "checkpoints/"


def resolve_checkpoint_candidates(*, checkpoint_root: Path, experiment_name: str, model_type: str = "") -> list[dict[str, Any]]:
    if not checkpoint_root.exists():
        return []
    needle = _normalize_name(experiment_name)
    candidates: list[dict[str, Any]] = []

    for path in checkpoint_root.rglob("*.pth"):
        stem = path.stem
        # Filter by name match
        if needle and needle not in _normalize_name(stem):
            continue
        # Filter by model type match
        if model_type and _normalize_name(model_type) not in _normalize_name(stem):
            continue

        stat = path.stat()
        mtime = stat.st_mtime
        size_mb = stat.st_size / (1024 * 1024)

        candidates.append({
            "path": path,
            "filename": path.name,
            "size_mb": size_mb,
            "mtime": mtime,
            "date": mtime
        })

    # Sort: Newest first, prioritize 'best' checkpoints
    candidates.sort(
        key=lambda item: (
            0 if "best" in item["filename"].lower() else 1,
            -item["mtime"]
        )
    )
    return candidates


def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())

