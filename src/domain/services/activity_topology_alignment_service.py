"""Deterministic activity-to-topology alignment checks for stats producers."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.process_structure import ProcessStructureDTO


LOGGABLE_BPMN_TAGS = {
    "userTask",
    "manualTask",
    "serviceTask",
    "scriptTask",
    "businessRuleTask",
    "sendTask",
    "receiveTask",
    "callActivity",
    "subProcess",
    "task",
}

STRUCTURAL_ONLY_BPMN_TAGS = {
    "startEvent",
    "endEvent",
    "exclusiveGateway",
    "parallelGateway",
    "inclusiveGateway",
    "eventBasedGateway",
    "complexGateway",
    "boundaryEvent",
    "intermediateCatchEvent",
    "intermediateThrowEvent",
}

_CLASSIFIER_SUFFIX_RE = re.compile(
    r"(\+|::)(complete|start|schedule|assign|withdraw|resume|suspend)$",
    re.IGNORECASE,
)
_SEPARATOR_RE = re.compile(r"[\s_\-]+")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class AlignmentGateConfig:
    profile: str = "legacy_exact"
    min_event_match_ratio: float = 0.6
    min_unique_activity_coverage: float = 0.6
    min_node_coverage: float = 0.0
    candidate_node_fields: tuple[str, ...] = ("id",)
    ignore_structural_only_nodes: bool = False
    strip_classifier_suffix: bool = False
    normalize_case: bool = False
    collapse_separators: bool = False
    fail_on_ambiguity: bool = True
    _profile_defaults_applied: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def for_profile(cls, profile: str) -> "AlignmentGateConfig":
        key = str(profile or "legacy_exact").strip() or "legacy_exact"
        if key == "legacy_exact":
            return cls(profile=key, _profile_defaults_applied=True)
        if key == "safe_normalized":
            return cls(
                profile=key,
                candidate_node_fields=("id", "name"),
                ignore_structural_only_nodes=True,
                strip_classifier_suffix=True,
                normalize_case=True,
                collapse_separators=True,
                fail_on_ambiguity=True,
                _profile_defaults_applied=True,
            )
        if key == "research_strict":
            return cls(
                profile=key,
                min_event_match_ratio=0.9,
                min_unique_activity_coverage=0.9,
                min_node_coverage=0.8,
                candidate_node_fields=("id", "name"),
                ignore_structural_only_nodes=True,
                strip_classifier_suffix=True,
                normalize_case=True,
                collapse_separators=True,
                fail_on_ambiguity=True,
                _profile_defaults_applied=True,
            )
        raise ValueError(
            "sync_stats.alignment_gate.profile must be one of "
            "{'legacy_exact', 'safe_normalized', 'research_strict'}."
        )

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "AlignmentGateConfig":
        raw = dict(payload or {})
        profile = str(raw.get("profile", raw.get("match_profile", "legacy_exact"))).strip()
        base = cls.for_profile(profile or "legacy_exact")
        fields_raw = raw.get("candidate_node_fields", base.candidate_node_fields)
        if isinstance(fields_raw, str):
            candidate_fields = tuple(
                item.strip() for item in fields_raw.split(",") if item.strip()
            )
        elif isinstance(fields_raw, Iterable):
            candidate_fields = tuple(
                str(item).strip() for item in fields_raw if str(item).strip()
            )
        else:
            candidate_fields = base.candidate_node_fields
        return replace(
            base,
            min_event_match_ratio=float(
                raw.get("min_event_match_ratio", base.min_event_match_ratio)
            ),
            min_unique_activity_coverage=float(
                raw.get(
                    "min_unique_activity_coverage",
                    base.min_unique_activity_coverage,
                )
            ),
            min_node_coverage=float(raw.get("min_node_coverage", base.min_node_coverage)),
            candidate_node_fields=candidate_fields or base.candidate_node_fields,
            ignore_structural_only_nodes=_to_bool(
                raw.get(
                    "ignore_structural_only_nodes",
                    base.ignore_structural_only_nodes,
                )
            ),
            strip_classifier_suffix=_to_bool(
                raw.get("strip_classifier_suffix", base.strip_classifier_suffix)
            ),
            normalize_case=_to_bool(raw.get("normalize_case", base.normalize_case)),
            collapse_separators=_to_bool(
                raw.get("collapse_separators", base.collapse_separators)
            ),
            fail_on_ambiguity=_to_bool(raw.get("fail_on_ambiguity", base.fail_on_ambiguity)),
            _profile_defaults_applied=True,
        )

    def with_profile_defaults(self) -> "AlignmentGateConfig":
        if self._profile_defaults_applied:
            return self
        base = self.for_profile(self.profile)
        return replace(
            base,
            min_event_match_ratio=self.min_event_match_ratio,
            min_unique_activity_coverage=self.min_unique_activity_coverage,
            min_node_coverage=self.min_node_coverage,
            _profile_defaults_applied=True,
        )


@dataclass(frozen=True)
class AlignmentSummary:
    profile: str
    scope_used: str
    event_count: int
    unique_event_activity_count: int
    structure_node_count: int
    loggable_node_count: int
    ignored_structural_node_count: int
    matched_event_count: int
    matched_unique_activity_count: int
    matched_loggable_node_count: int
    event_match_ratio: float
    unique_activity_coverage: float
    node_coverage: float
    min_event_match_ratio: float
    min_unique_activity_coverage: float
    min_node_coverage: float
    is_aligned: bool
    alignment_reason: str
    alignment_failures: list[str] = field(default_factory=list)
    unmatched_event_activities_top: list[str] = field(default_factory=list)
    ambiguous_event_activities_top: list[str] = field(default_factory=list)
    match_counts_by_strategy: dict[str, int] = field(default_factory=dict)
    candidate_node_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "scope_used": self.scope_used,
            "event_count": self.event_count,
            "unique_event_activity_count": self.unique_event_activity_count,
            "structure_node_count": self.structure_node_count,
            "loggable_node_count": self.loggable_node_count,
            "ignored_structural_node_count": self.ignored_structural_node_count,
            "matched_event_count": self.matched_event_count,
            "matched_unique_activity_count": self.matched_unique_activity_count,
            "matched_loggable_node_count": self.matched_loggable_node_count,
            "event_match_ratio": self.event_match_ratio,
            "unique_activity_coverage": self.unique_activity_coverage,
            "node_coverage": self.node_coverage,
            "min_event_match_ratio": self.min_event_match_ratio,
            "min_unique_activity_coverage": self.min_unique_activity_coverage,
            "min_node_coverage": self.min_node_coverage,
            "is_aligned": self.is_aligned,
            "alignment_reason": self.alignment_reason,
            "alignment_failures": list(self.alignment_failures),
            "unmatched_event_activities_top": list(self.unmatched_event_activities_top),
            "ambiguous_event_activities_top": list(self.ambiguous_event_activities_top),
            "match_counts_by_strategy": dict(self.match_counts_by_strategy),
            "candidate_node_fields": list(self.candidate_node_fields),
        }


@dataclass(frozen=True)
class _Candidate:
    node_id: str
    field: str
    value: str
    normalized: str
    is_loggable: bool


@dataclass(frozen=True)
class _Match:
    node_id: str | None
    strategy: str
    ambiguous: bool = False


class ActivityTopologyAlignmentService:
    """Evaluate whether event activities are explainably aligned to topology nodes."""

    def evaluate(
        self,
        *,
        events: Sequence[ProcessEventDTO],
        dto: ProcessStructureDTO,
        config: AlignmentGateConfig,
        scope_used: str,
    ) -> AlignmentSummary:
        cfg = config.with_profile_defaults()
        candidates = self._candidates(dto, cfg)
        node_ids = {candidate.node_id for candidate in candidates}
        loggable_node_ids = {
            candidate.node_id for candidate in candidates if candidate.is_loggable
        }
        if not cfg.ignore_structural_only_nodes:
            loggable_node_ids = set(node_ids)

        event_activity_ids = [
            self._clean(event.activity_def_id, cfg)
            for event in events
            if self._clean(event.activity_def_id, cfg)
        ]
        event_count = len(event_activity_ids)
        unique_event_activity_ids = set(event_activity_ids)
        unique_event_count = len(unique_event_activity_ids)

        matched_event_count = 0
        matched_unique_event_ids: set[str] = set()
        matched_loggable_node_ids: set[str] = set()
        unmatched_counter: Counter[str] = Counter()
        ambiguous_counter: Counter[str] = Counter()
        strategy_counter: Counter[str] = Counter(
            {
                "exact_id": 0,
                "exact_name": 0,
                "normalized_id": 0,
                "normalized_name": 0,
            }
        )

        for activity_id in event_activity_ids:
            match = self._match_activity(activity_id, candidates, cfg)
            if match.ambiguous:
                ambiguous_counter[activity_id] += 1
                continue
            if match.node_id:
                matched_event_count += 1
                matched_unique_event_ids.add(activity_id)
                strategy_counter[match.strategy] += 1
                if match.node_id in loggable_node_ids:
                    matched_loggable_node_ids.add(match.node_id)
                continue
            unmatched_counter[activity_id] += 1

        structure_node_count = len(node_ids)
        loggable_node_count = len(loggable_node_ids)
        ignored_structural_node_count = max(0, structure_node_count - loggable_node_count)
        matched_unique_count = len(matched_unique_event_ids)
        matched_loggable_count = len(matched_loggable_node_ids)
        event_match_ratio = _safe_ratio(matched_event_count, event_count)
        unique_activity_coverage = _safe_ratio(matched_unique_count, unique_event_count)
        node_coverage = _safe_ratio(matched_loggable_count, loggable_node_count)

        failures: list[str] = []
        if structure_node_count <= 0:
            failures.append("empty_structure_nodes")
        if event_count <= 0:
            failures.append("no_events_for_alignment")
        if ambiguous_counter and cfg.fail_on_ambiguity:
            failures.append("ambiguous_activity_mapping")
        if event_count > 0 and event_match_ratio < cfg.min_event_match_ratio:
            failures.append("below_min_event_match_ratio")
        if (
            unique_event_count > 0
            and unique_activity_coverage < cfg.min_unique_activity_coverage
        ):
            failures.append("below_min_unique_activity_coverage")
        if loggable_node_count > 0 and node_coverage < cfg.min_node_coverage:
            failures.append("below_min_node_coverage")

        unique_failures = list(dict.fromkeys(failures))
        return AlignmentSummary(
            profile=cfg.profile,
            scope_used=str(scope_used),
            event_count=event_count,
            unique_event_activity_count=unique_event_count,
            structure_node_count=structure_node_count,
            loggable_node_count=loggable_node_count,
            ignored_structural_node_count=ignored_structural_node_count,
            matched_event_count=matched_event_count,
            matched_unique_activity_count=matched_unique_count,
            matched_loggable_node_count=matched_loggable_count,
            event_match_ratio=float(event_match_ratio),
            unique_activity_coverage=float(unique_activity_coverage),
            node_coverage=float(node_coverage),
            min_event_match_ratio=float(cfg.min_event_match_ratio),
            min_unique_activity_coverage=float(cfg.min_unique_activity_coverage),
            min_node_coverage=float(cfg.min_node_coverage),
            is_aligned=not unique_failures,
            alignment_reason=unique_failures[0] if unique_failures else "ok",
            alignment_failures=unique_failures,
            unmatched_event_activities_top=[
                key for key, _ in unmatched_counter.most_common(10)
            ],
            ambiguous_event_activities_top=[
                key for key, _ in ambiguous_counter.most_common(10)
            ],
            match_counts_by_strategy=dict(strategy_counter),
            candidate_node_fields=list(cfg.candidate_node_fields),
        )

    def _candidates(
        self,
        dto: ProcessStructureDTO,
        cfg: AlignmentGateConfig,
    ) -> list[_Candidate]:
        nodes = list(dto.nodes or [])
        if not nodes:
            edge_ids = {node_id for edge in dto.allowed_edges for node_id in edge}
            nodes = [
                {
                    "id": node_id,
                    "bpmn_tag": "task",
                    "type": "task",
                    "activity_type": "task",
                }
                for node_id in sorted(edge_ids)
            ]

        candidates: list[_Candidate] = []
        for node in nodes:
            node_id = self._clean(node.get("id"), cfg)
            if not node_id:
                continue
            is_loggable = self._is_loggable(node, cfg)
            for field_name in cfg.candidate_node_fields:
                value = self._clean(node.get(field_name), cfg)
                if not value:
                    continue
                candidates.append(
                    _Candidate(
                        node_id=node_id,
                        field=str(field_name),
                        value=value,
                        normalized=self._canonicalize(value, cfg),
                        is_loggable=is_loggable,
                    )
                )
        return candidates

    def _match_activity(
        self,
        activity_id: str,
        candidates: Sequence[_Candidate],
        cfg: AlignmentGateConfig,
    ) -> _Match:
        exact_id = self._lookup(
            candidates,
            field="id",
            value=activity_id,
            normalized=False,
            cfg=cfg,
        )
        if exact_id.ambiguous or exact_id.node_id:
            return exact_id

        if "name" in cfg.candidate_node_fields:
            normalized_name = self._lookup(
                candidates,
                field="name",
                value=activity_id,
                normalized=True,
                cfg=cfg,
            )
            if normalized_name.ambiguous and cfg.fail_on_ambiguity:
                return normalized_name
            exact_name = self._lookup(
                candidates,
                field="name",
                value=activity_id,
                normalized=False,
                cfg=cfg,
            )
            if exact_name.ambiguous or exact_name.node_id:
                return exact_name

        normalized_id = self._lookup(
            candidates,
            field="id",
            value=activity_id,
            normalized=True,
            cfg=cfg,
        )
        if normalized_id.ambiguous or normalized_id.node_id:
            return normalized_id

        if "name" in cfg.candidate_node_fields:
            return self._lookup(
                candidates,
                field="name",
                value=activity_id,
                normalized=True,
                cfg=cfg,
            )
        return _Match(node_id=None, strategy="unmatched")

    def _lookup(
        self,
        candidates: Sequence[_Candidate],
        *,
        field: str,
        value: str,
        normalized: bool,
        cfg: AlignmentGateConfig,
    ) -> _Match:
        key = self._canonicalize(value, cfg)
        matches = [
            candidate
            for candidate in candidates
            if candidate.field == field
            and ((candidate.normalized == key) if normalized else (candidate.value == value))
        ]
        node_ids = sorted({candidate.node_id for candidate in matches})
        strategy_prefix = "normalized" if normalized else "exact"
        strategy = f"{strategy_prefix}_{field}"
        if len(node_ids) > 1:
            return _Match(node_id=None, strategy=strategy, ambiguous=True)
        if node_ids:
            return _Match(node_id=node_ids[0], strategy=strategy)
        return _Match(node_id=None, strategy=strategy)

    def _is_loggable(self, node: dict[str, Any], cfg: AlignmentGateConfig) -> bool:
        if not cfg.ignore_structural_only_nodes:
            return True
        tag = str(
            node.get("bpmn_tag") or node.get("activity_type") or node.get("type") or ""
        ).strip()
        if tag in LOGGABLE_BPMN_TAGS:
            return True
        if tag in STRUCTURAL_ONLY_BPMN_TAGS:
            return False
        return True

    def _clean(self, value: Any, cfg: AlignmentGateConfig) -> str:
        text = str(value or "").strip()
        if cfg.strip_classifier_suffix:
            text = _CLASSIFIER_SUFFIX_RE.sub("", text).strip()
        return text

    def _canonicalize(self, value: str, cfg: AlignmentGateConfig) -> str:
        text = self._clean(value, cfg)
        if cfg.normalize_case:
            text = text.lower()
        if cfg.collapse_separators:
            text = _SEPARATOR_RE.sub("", text)
        return text


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
