from __future__ import annotations

from pathlib import Path


DEFAULT_CHECKPOINT_VALUE = "checkpoints/"


def resolve_checkpoint_candidates(*, checkpoint_root: Path, experiment_name: str) -> list[Path]:
    if not experiment_name.strip() or not checkpoint_root.exists():
        return []
    needle = _normalize_name(experiment_name)
    candidates: list[Path] = []
    for path in checkpoint_root.rglob("*.pth"):
        if needle in _normalize_name(path.stem):
            candidates.append(path)
    return sorted(candidates, key=lambda item: (0 if "best" in item.stem.lower() else 1, str(item).lower()))


def _normalize_name(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())

