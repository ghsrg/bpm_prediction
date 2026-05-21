"""Deterministic sampling policy for selective structural traces."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Iterable


def classify_trace_reason(
    *,
    strict_correct: bool,
    pred_in_mask: bool | None,
    target_in_mask: bool | None,
    confidence: float,
    version_seen: bool,
    high_confidence_threshold: float = 0.8,
    low_confidence_threshold: float = 0.4,
) -> str | None:
    """Classify why a prediction is worth tracing."""
    if pred_in_mask is False:
        return "oos_prediction"
    if (not strict_correct) and pred_in_mask is True and target_in_mask is True:
        return "strict_error_but_allowed"
    if (not strict_correct) and float(confidence) >= float(high_confidence_threshold):
        return "high_confidence_error"
    if strict_correct and float(confidence) <= float(low_confidence_threshold):
        return "low_confidence_correct"
    if not version_seen:
        return "version_first"
    return None


@dataclass
class TraceSamplingPolicy:
    """Bounded deterministic sampler for trace events."""

    enabled: bool = False
    stages: set[str] = field(default_factory=set)
    sample_policy: str = "interesting"
    max_traces_per_run: int = 80
    max_traces_per_stage: int = 40
    max_traces_per_version: int = 10
    random_sample_rate: float = 0.0
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.4
    top_k: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        self.max_traces_per_run = max(0, int(self.max_traces_per_run))
        self.max_traces_per_stage = max(0, int(self.max_traces_per_stage))
        self.max_traces_per_version = max(0, int(self.max_traces_per_version))
        self.random_sample_rate = min(1.0, max(0.0, float(self.random_sample_rate)))
        self.top_k = max(1, int(self.top_k))
        self.sample_policy = str(self.sample_policy or "interesting").strip().lower()
        self._run_count = 0
        self._stage_counts: dict[str, int] = {}
        self._version_counts: dict[str, int] = {}
        self._rng = random.Random(int(self.seed))

    @classmethod
    def from_config(cls, config: dict[str, Any] | None, *, seed: int = 42) -> "TraceSamplingPolicy":
        cfg = dict(config or {})
        raw_stages = cfg.get("stages", [])
        stages = cls._parse_stages(raw_stages)
        return cls(
            enabled=_as_bool(cfg.get("enabled"), default=False),
            stages=stages,
            sample_policy=str(cfg.get("sample_policy", "interesting")),
            max_traces_per_run=int(cfg.get("max_traces_per_run", 80)),
            max_traces_per_stage=int(cfg.get("max_traces_per_stage", 40)),
            max_traces_per_version=int(cfg.get("max_traces_per_version", 10)),
            random_sample_rate=float(cfg.get("random_sample_rate", 0.0)),
            high_confidence_threshold=float(cfg.get("high_confidence_threshold", 0.8)),
            low_confidence_threshold=float(cfg.get("low_confidence_threshold", 0.4)),
            top_k=int(cfg.get("top_k", 5)),
            seed=int(cfg.get("seed", seed)),
        )

    @staticmethod
    def _parse_stages(raw: Any) -> set[str]:
        if raw is None:
            return set()
        if isinstance(raw, str):
            return {item.strip() for item in raw.split(",") if item.strip()}
        if isinstance(raw, Iterable):
            return {str(item).strip() for item in raw if str(item).strip()}
        return set()

    def should_record(self, *, stage: str, version: str, reason: str | None) -> bool:
        if not self.enabled:
            return False
        stage_key = str(stage or "").strip()
        version_key = str(version or "__unknown__").strip() or "__unknown__"
        reason_key = str(reason or "").strip()
        if self.stages and stage_key not in self.stages:
            return False
        if self._run_count >= self.max_traces_per_run:
            return False
        if int(self._stage_counts.get(stage_key, 0)) >= self.max_traces_per_stage:
            return False
        if int(self._version_counts.get(version_key, 0)) >= self.max_traces_per_version:
            return False
        if not reason_key:
            if self.random_sample_rate <= 0.0:
                return False
            if self._rng.random() > self.random_sample_rate:
                return False
        elif self.sample_policy not in {"interesting", "all"} and reason_key != "random":
            return False

        self._run_count += 1
        self._stage_counts[stage_key] = int(self._stage_counts.get(stage_key, 0)) + 1
        self._version_counts[version_key] = int(self._version_counts.get(version_key, 0)) + 1
        return True

    def version_seen(self, version: str) -> bool:
        version_key = str(version or "__unknown__").strip() or "__unknown__"
        return int(self._version_counts.get(version_key, 0)) > 0


def _as_bool(raw: Any, *, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)
