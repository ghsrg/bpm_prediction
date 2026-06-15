"""Deterministic aliases for topology-native candidate label matching."""

from __future__ import annotations

import re

_XES_LIFECYCLE_SUFFIXES = {
    "complete",
    "start",
    "ate_abort",
    "pi_abort",
    "manualskip",
    "autoskip",
}


def _canonical_label(value: str) -> str:
    canonical = re.sub(r"[^\w]+", "_", value.strip().lower(), flags=re.UNICODE)
    return re.sub(r"_+", "_", canonical).strip("_")


def candidate_label_aliases(value: object) -> set[str]:
    """Return conservative aliases for candidate/target label matching.

    Exact labels remain the primary contract. Aliases only handle deterministic
    formatting differences introduced by event-log lifecycle classifiers, e.g.
    ``W_Task+COMPLETE`` in XES targets versus ``W_Task`` in topology nodes.
    """

    raw = str(value).strip()
    if not raw:
        return set()
    aliases = {raw}
    if "+" in raw:
        base, suffix = raw.rsplit("+", 1)
        if suffix.strip().lower() in _XES_LIFECYCLE_SUFFIXES:
            base = base.strip()
            if base:
                aliases.add(base)
    normalized = {_canonical_label(item) for item in aliases}
    aliases.update(item for item in normalized if item)
    return aliases


def candidate_label_metric_key(value: object) -> str:
    """Return deterministic label key for stable candidate-space metrics."""

    raw = str(value).strip()
    if not raw:
        return ""
    if "+" in raw:
        base, suffix = raw.rsplit("+", 1)
        if suffix.strip().lower() in _XES_LIFECYCLE_SUFFIXES and base.strip():
            raw = base.strip()
    return raw
