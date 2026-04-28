# Activity-to-Topology Alignment Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the P0 `activity_to_topology_alignment_gate` debt by making `sync-stats` alignment deterministic, auditable, research-grade, and covered by pure-domain plus real-object tests.

**Architecture:** Extract the alignment logic from `tools/sync_stats.py` into a pure domain service. Keep `sync-stats` as orchestration only: config parsing, event loading, repository writes, and summary output. Preserve current operational policies (`write_with_flag`, `skip_snapshot`, `raise`) while adding explicit matching profiles, loggable-node denominator semantics, ambiguity diagnostics, and a locked research profile.

**Tech Stack:** Python 3.11 in project `.venv`, Pydantic DTOs, `pytest`, existing `FileBasedKnowledgeGraphRepository`, existing `sync_stats_main`, existing BPMN/XES integration test fixtures.

---

## Current Verified Context

- Current debt: `docs/current/architecture-debt.md` -> `activity_to_topology_alignment_gate`.
- Target decision: `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`.
- Current implementation: `tools/sync_stats.py`.
- Current gate defaults:
  - `min_event_match_ratio = 0.6`
  - `min_unique_activity_coverage = 0.6`
  - `min_node_coverage = 0.0`
  - `on_fail = write_with_flag`
  - `warn_on_fail = true`
- Current weakness:
  - `_dto_activity_ids(dto)` uses `dto.nodes[].id` or `allowed_edges`.
  - `_build_alignment_summary(...)` exact-matches normalized `event.activity_def_id` against node ids only.
  - `node_coverage` denominator includes structural-only BPMN nodes when present.
  - Diagnostics do not show match provenance or ambiguity.
- Existing tests already cover:
  - `skip_snapshot` on mismatch in `tests/application/test_sync_stats_tool.py`.
  - `write_with_flag` alignment metadata in `tests/application/test_sync_stats_tool.py`.
  - BPMN -> XES -> stats -> GNN smoke in `tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py`.
- `configs/ui/config_catalog.yaml` already contains current `sync_stats.alignment_gate.*` keys as of 2026-04-27.

---

## Design Decision

Implement the first closure pass with three explicit profiles:

1. `legacy_exact`
   - Preserves current exact-id behavior for comparison and rollback.
   - Uses all DTO ids only.
   - No normalization beyond trim.

2. `safe_normalized`
   - Intended practical default after tests prove behavior.
   - Uses loggable nodes only.
   - Matches by exact id, exact unique name, normalized id, normalized unique name.
   - Supports optional classifier suffix stripping.
   - Fails closed on ambiguity.

3. `research_strict`
   - Same conservative matching semantics as `safe_normalized`.
   - Uses strict thresholds and `on_fail=raise` in research configs.
   - Does not allow `write_with_flag` snapshots as final evidence.

Do not introduce fuzzy matching, edit distance, or automatic global `Task_` prefix stripping in this task. These create false-positive risk and are harder to defend.

---

## File Structure

### Create

- `src/domain/services/activity_topology_alignment_service.py`
  - Pure alignment service.
  - Owns matching profiles, canonicalization, candidate index, metrics, and diagnostics.

- `tests/domain/test_activity_topology_alignment_service.py`
  - Pure tests for candidate filtering, matching provenance, ambiguity, and denominator semantics.

- `tests/application/test_sync_stats_alignment_gate_real_objects.py`
  - Real-object `sync_stats_main` tests using `FileBasedKnowledgeGraphRepository`, real `ProcessStructureDTO`, generated XES, and persisted metadata.

### Modify

- `tools/sync_stats.py`
  - Replace `_dto_activity_ids(...)` and `_build_alignment_summary(...)` usage with domain service.
  - Keep config parsing and `on_fail` behavior in tool.

- `configs/ui/config_catalog.yaml`
  - Add new optional profile/policy fields introduced by this implementation.

- `docs/current/architecture-debt.md`
  - Mark `activity_to_topology_alignment_gate` as closed only after tests pass and docs are updated.

- `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`
  - Change `Status: Proposed` to `Accepted` only after implementation and tests pass.

- `docs/LLD_MVP2_5.MD`
  - Document matching profiles, denominator semantics, and ambiguity behavior.

- `docs/DATA_MODEL_MVP2_5.MD`
  - Document added alignment diagnostic fields in `metadata.stats_contract.alignment`.

- `docs/runbooks/mvp2_5-commands.md`
  - Add research-grade alignment config snippet.

---

## Target Alignment Contract

### Metrics

- `event_match_ratio = matched_event_count / event_count`
- `unique_activity_coverage = matched_unique_event_activity_count / unique_event_activity_count`
- `node_coverage = matched_loggable_node_count / loggable_node_count`

### Loggable Node Denominator

Loggable node types:

```python
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
```

Structural-only nodes excluded from denominator by default:

```python
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
```

### Required Diagnostics

`metadata.stats_contract.alignment` must include at least:

- `profile`
- `scope_used`
- `event_count`
- `unique_event_activity_count`
- `structure_node_count`
- `loggable_node_count`
- `matched_event_count`
- `matched_unique_activity_count`
- `matched_loggable_node_count`
- `event_match_ratio`
- `unique_activity_coverage`
- `node_coverage`
- `is_aligned`
- `alignment_reason`
- `alignment_failures`
- `unmatched_event_activities_top`
- `ambiguous_event_activities_top`
- `match_counts_by_strategy`
- `candidate_node_fields`
- `ignored_structural_node_count`

---

## Task 1: Add Pure Domain Service Tests First

**Files:**
- Create: `tests/domain/test_activity_topology_alignment_service.py`
- Create later in Task 2: `src/domain/services/activity_topology_alignment_service.py`

- [x] **Step 1: Write failing tests for exact id, name fallback, normalized fallback, structural denominator, and ambiguity**

Create `tests/domain/test_activity_topology_alignment_service.py`:

```python
from __future__ import annotations

import pytest

from src.domain.entities.process_event import ProcessEventDTO
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.services.activity_topology_alignment_service import (
    AlignmentGateConfig,
    ActivityTopologyAlignmentService,
)


def _event(activity: str, name: str | None = None) -> ProcessEventDTO:
    return ProcessEventDTO(
        case_id="C1",
        activity_def_id=activity,
        activity_name=name,
        proc_def_key="loan_proc",
        proc_def_version="v1",
    )


def _dto(nodes: list[dict], edges: list[tuple[str, str]] | None = None) -> ProcessStructureDTO:
    return ProcessStructureDTO(
        version="v1",
        allowed_edges=edges or [("TaskA", "TaskB")],
        nodes=nodes,
    )


def test_exact_id_profile_preserves_legacy_matching():
    dto = _dto(
        [
            {"id": "TaskA", "name": "Task A", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "TaskB", "name": "Task B", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
        ],
        [("TaskA", "TaskB")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("TaskA"), _event("TaskB")],
        dto=dto,
        config=AlignmentGateConfig(profile="legacy_exact", min_event_match_ratio=1.0, min_unique_activity_coverage=1.0, min_node_coverage=1.0),
        scope_used="version",
    )

    assert summary.is_aligned is True
    assert summary.event_match_ratio == pytest.approx(1.0)
    assert summary.node_coverage == pytest.approx(1.0)
    assert summary.match_counts_by_strategy["exact_id"] == 2


def test_safe_profile_matches_unique_node_name_when_id_differs():
    dto = _dto(
        [
            {"id": "Task_Approve", "name": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "Task_Archive", "name": "Archive", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
        ],
        [("Task_Approve", "Task_Archive")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("Approve"), _event("Archive")],
        dto=dto,
        config=AlignmentGateConfig(profile="safe_normalized", min_event_match_ratio=1.0, min_unique_activity_coverage=1.0, min_node_coverage=1.0),
        scope_used="version",
    )

    assert summary.is_aligned is True
    assert summary.event_match_ratio == pytest.approx(1.0)
    assert summary.match_counts_by_strategy["exact_name"] == 2


def test_safe_profile_normalizes_case_and_separators():
    dto = _dto(
        [
            {"id": "task_a", "name": "Task A", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "task_b", "name": "Task B", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
        ],
        [("task_a", "task_b")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("Task-A"), _event("TASK B")],
        dto=dto,
        config=AlignmentGateConfig(profile="safe_normalized", min_event_match_ratio=1.0, min_unique_activity_coverage=1.0, min_node_coverage=1.0),
        scope_used="version",
    )

    assert summary.is_aligned is True
    assert summary.match_counts_by_strategy["normalized_id"] + summary.match_counts_by_strategy["normalized_name"] == 2


def test_node_coverage_ignores_structural_only_nodes():
    dto = _dto(
        [
            {"id": "StartEvent", "name": "Start", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
            {"id": "TaskA", "name": "Task A", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "GatewayChoice", "name": "Choice", "bpmn_tag": "exclusiveGateway", "type": "exclusiveGateway", "activity_type": "exclusiveGateway"},
            {"id": "TaskB", "name": "Task B", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
            {"id": "EndEvent", "name": "End", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
        ],
        [("StartEvent", "TaskA"), ("TaskA", "GatewayChoice"), ("GatewayChoice", "TaskB"), ("TaskB", "EndEvent")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("TaskA"), _event("TaskB")],
        dto=dto,
        config=AlignmentGateConfig(profile="safe_normalized", min_event_match_ratio=1.0, min_unique_activity_coverage=1.0, min_node_coverage=1.0),
        scope_used="version",
    )

    assert summary.is_aligned is True
    assert summary.structure_node_count == 5
    assert summary.loggable_node_count == 2
    assert summary.ignored_structural_node_count == 3
    assert summary.node_coverage == pytest.approx(1.0)


def test_ambiguous_normalized_name_fails_closed():
    dto = _dto(
        [
            {"id": "TaskA1", "name": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "TaskA2", "name": "approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
        ],
        [("TaskA1", "TaskA2")],
    )

    summary = ActivityTopologyAlignmentService().evaluate(
        events=[_event("Approve")],
        dto=dto,
        config=AlignmentGateConfig(profile="safe_normalized", min_event_match_ratio=1.0, min_unique_activity_coverage=1.0, min_node_coverage=0.0),
        scope_used="version",
    )

    assert summary.is_aligned is False
    assert "ambiguous_activity_mapping" in summary.alignment_failures
    assert summary.ambiguous_event_activities_top == ["Approve"]
```

- [x] **Step 2: Run tests and verify they fail because service does not exist**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_activity_topology_alignment_service.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'src.domain.services.activity_topology_alignment_service'
```

---

## Task 2: Implement Pure Alignment Service

**Files:**
- Create: `src/domain/services/activity_topology_alignment_service.py`
- Test: `tests/domain/test_activity_topology_alignment_service.py`

- [x] **Step 1: Create service with profiles, canonicalization, candidate index, and diagnostics**

Create `src/domain/services/activity_topology_alignment_service.py`:

```python
"""Activity-to-topology alignment service for sync-stats research safety."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
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

_SEPARATOR_RE = re.compile(r"[\s_\-]+")
_CLASSIFIER_SUFFIX_RE = re.compile(r"(\+|::)(complete|start|schedule|assign|withdraw|resume|suspend)$", re.IGNORECASE)


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

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "AlignmentGateConfig":
        raw = dict(payload or {})
        profile = str(raw.get("profile", raw.get("match_profile", "legacy_exact"))).strip() or "legacy_exact"
        base = cls.for_profile(profile)
        fields_raw = raw.get("candidate_node_fields", base.candidate_node_fields)
        if isinstance(fields_raw, str):
            fields = tuple(item.strip() for item in fields_raw.split(",") if item.strip())
        elif isinstance(fields_raw, Iterable):
            fields = tuple(str(item).strip() for item in fields_raw if str(item).strip())
        else:
            fields = base.candidate_node_fields
        return cls(
            profile=profile,
            min_event_match_ratio=float(raw.get("min_event_match_ratio", base.min_event_match_ratio)),
            min_unique_activity_coverage=float(raw.get("min_unique_activity_coverage", base.min_unique_activity_coverage)),
            min_node_coverage=float(raw.get("min_node_coverage", base.min_node_coverage)),
            candidate_node_fields=fields or base.candidate_node_fields,
            ignore_structural_only_nodes=bool(raw.get("ignore_structural_only_nodes", base.ignore_structural_only_nodes)),
            strip_classifier_suffix=bool(raw.get("strip_classifier_suffix", base.strip_classifier_suffix)),
            normalize_case=bool(raw.get("normalize_case", base.normalize_case)),
            collapse_separators=bool(raw.get("collapse_separators", base.collapse_separators)),
            fail_on_ambiguity=bool(raw.get("fail_on_ambiguity", base.fail_on_ambiguity)),
        )

    @classmethod
    def for_profile(cls, profile: str) -> "AlignmentGateConfig":
        key = str(profile).strip() or "legacy_exact"
        if key == "legacy_exact":
            return cls(profile=key)
        if key == "safe_normalized":
            return cls(
                profile=key,
                candidate_node_fields=("id", "name"),
                ignore_structural_only_nodes=True,
                strip_classifier_suffix=True,
                normalize_case=True,
                collapse_separators=True,
                fail_on_ambiguity=True,
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
            )
        raise ValueError("sync_stats.alignment_gate.profile must be one of {'legacy_exact', 'safe_normalized', 'research_strict'}.")


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


class ActivityTopologyAlignmentService:
    def evaluate(
        self,
        *,
        events: Sequence[ProcessEventDTO],
        dto: ProcessStructureDTO,
        config: AlignmentGateConfig,
        scope_used: str,
    ) -> AlignmentSummary:
        candidates, structure_node_count, loggable_node_count = self._candidates(dto, config)
        exact_id = self._index(candidates, field="id", normalized=False)
        exact_name = self._index(candidates, field="name", normalized=False)
        normalized_id = self._index(candidates, field="id", normalized=True)
        normalized_name = self._index(candidates, field="name", normalized=True)

        event_values = [self._clean(event.activity_def_id) for event in events if self._clean(event.activity_def_id)]
        unique_event_values = set(event_values)
        matched_events: list[str] = []
        matched_nodes: set[str] = set()
        unmatched_counter: Counter[str] = Counter()
        ambiguous_counter: Counter[str] = Counter()
        strategy_counts: Counter[str] = Counter()

        for activity in event_values:
            match = self._match_activity(
                activity=activity,
                config=config,
                exact_id=exact_id,
                exact_name=exact_name,
                normalized_id=normalized_id,
                normalized_name=normalized_name,
            )
            if match["ambiguous"]:
                ambiguous_counter[activity] += 1
                unmatched_counter[activity] += 1
                continue
            node_id = str(match["node_id"])
            if node_id:
                matched_events.append(activity)
                matched_nodes.add(node_id)
                strategy_counts[str(match["strategy"])] += 1
            else:
                unmatched_counter[activity] += 1

        event_count = len(event_values)
        unique_event_count = len(unique_event_values)
        matched_event_count = len(matched_events)
        matched_unique_count = len({item for item in unique_event_values if item not in unmatched_counter})
        matched_loggable_node_count = len(matched_nodes)

        event_match_ratio = self._safe_ratio(matched_event_count, event_count)
        unique_activity_coverage = self._safe_ratio(matched_unique_count, unique_event_count)
        node_coverage = self._safe_ratio(matched_loggable_node_count, loggable_node_count)

        failures: list[str] = []
        if loggable_node_count <= 0:
            failures.append("empty_structure_nodes")
        if event_count <= 0:
            failures.append("no_events_for_alignment")
        if config.fail_on_ambiguity and ambiguous_counter:
            failures.append("ambiguous_activity_mapping")
        if event_count > 0 and event_match_ratio < config.min_event_match_ratio:
            failures.append("below_min_event_match_ratio")
        if unique_event_count > 0 and unique_activity_coverage < config.min_unique_activity_coverage:
            failures.append("below_min_unique_activity_coverage")
        if loggable_node_count > 0 and node_coverage < config.min_node_coverage:
            failures.append("below_min_node_coverage")

        reason = failures[0] if failures else "ok"
        return AlignmentSummary(
            profile=config.profile,
            scope_used=str(scope_used),
            event_count=event_count,
            unique_event_activity_count=unique_event_count,
            structure_node_count=structure_node_count,
            loggable_node_count=loggable_node_count,
            ignored_structural_node_count=max(0, structure_node_count - loggable_node_count),
            matched_event_count=matched_event_count,
            matched_unique_activity_count=matched_unique_count,
            matched_loggable_node_count=matched_loggable_node_count,
            event_match_ratio=event_match_ratio,
            unique_activity_coverage=unique_activity_coverage,
            node_coverage=node_coverage,
            min_event_match_ratio=float(config.min_event_match_ratio),
            min_unique_activity_coverage=float(config.min_unique_activity_coverage),
            min_node_coverage=float(config.min_node_coverage),
            is_aligned=not failures,
            alignment_reason=reason,
            alignment_failures=list(dict.fromkeys(failures)),
            unmatched_event_activities_top=[key for key, _ in unmatched_counter.most_common(10)],
            ambiguous_event_activities_top=[key for key, _ in ambiguous_counter.most_common(10)],
            match_counts_by_strategy={key: int(strategy_counts.get(key, 0)) for key in ["exact_id", "exact_name", "normalized_id", "normalized_name"]},
            candidate_node_fields=list(config.candidate_node_fields),
        )

    def _candidates(self, dto: ProcessStructureDTO, config: AlignmentGateConfig) -> tuple[list[_Candidate], int, int]:
        nodes = [node for node in (dto.nodes or []) if isinstance(node, dict)]
        if not nodes:
            fallback_nodes = sorted({src for src, _ in dto.allowed_edges} | {dst for _, dst in dto.allowed_edges})
            nodes = [{"id": node_id, "bpmn_tag": "task", "type": "task", "activity_type": "task"} for node_id in fallback_nodes]

        result: list[_Candidate] = []
        loggable_count = 0
        for node in nodes:
            node_id = self._clean(node.get("id"))
            if not node_id:
                continue
            is_loggable = self._is_loggable(node, config)
            if is_loggable:
                loggable_count += 1
            if config.ignore_structural_only_nodes and not is_loggable:
                continue
            for field_name in config.candidate_node_fields:
                raw_value = self._clean(node.get(field_name))
                if not raw_value:
                    continue
                result.append(
                    _Candidate(
                        node_id=node_id,
                        field=str(field_name),
                        value=raw_value,
                        normalized=self._canonicalize(raw_value, config),
                        is_loggable=is_loggable,
                    )
                )
        return result, len([node for node in nodes if self._clean(node.get("id"))]), loggable_count

    def _match_activity(self, *, activity: str, config: AlignmentGateConfig, exact_id: dict[str, set[str]], exact_name: dict[str, set[str]], normalized_id: dict[str, set[str]], normalized_name: dict[str, set[str]]) -> dict[str, Any]:
        checks = [
            ("exact_id", activity, exact_id),
            ("exact_name", activity, exact_name),
            ("normalized_id", self._canonicalize(activity, config), normalized_id),
            ("normalized_name", self._canonicalize(activity, config), normalized_name),
        ]
        if config.profile == "legacy_exact":
            checks = [("exact_id", activity, exact_id)]
        for strategy, key, index in checks:
            matches = index.get(key, set())
            if len(matches) == 1:
                return {"node_id": next(iter(matches)), "strategy": strategy, "ambiguous": False}
            if len(matches) > 1:
                return {"node_id": "", "strategy": strategy, "ambiguous": True}
        return {"node_id": "", "strategy": "", "ambiguous": False}

    @staticmethod
    def _index(candidates: Sequence[_Candidate], *, field: str, normalized: bool) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for candidate in candidates:
            if candidate.field != field:
                continue
            key = candidate.normalized if normalized else candidate.value
            result.setdefault(key, set()).add(candidate.node_id)
        return result

    @staticmethod
    def _is_loggable(node: dict[str, Any], config: AlignmentGateConfig) -> bool:
        if not config.ignore_structural_only_nodes:
            return True
        values = {
            str(node.get("bpmn_tag", "")).strip(),
            str(node.get("type", "")).strip(),
            str(node.get("activity_type", "")).strip(),
        }
        if values.intersection(LOGGABLE_BPMN_TAGS):
            return True
        if values.intersection(STRUCTURAL_ONLY_BPMN_TAGS):
            return False
        return True

    @classmethod
    def _canonicalize(cls, value: str, config: AlignmentGateConfig) -> str:
        text = cls._clean(value)
        if config.strip_classifier_suffix:
            text = _CLASSIFIER_SUFFIX_RE.sub("", text)
        if config.normalize_case:
            text = text.casefold()
        if config.collapse_separators:
            text = _SEPARATOR_RE.sub(" ", text).strip()
        return text

    @staticmethod
    def _clean(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _safe_ratio(numerator: int, denominator: int) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator / denominator)
```

- [x] **Step 2: Run pure service tests and fix only if the shown code has small mechanical issues**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_activity_topology_alignment_service.py -v
```

Expected:

```text
5 passed
```

---

## Task 3: Integrate Service Into `sync-stats` Without Changing Default Behavior

**Files:**
- Modify: `tools/sync_stats.py`
- Test: existing `tests/application/test_sync_stats_tool.py`

- [x] **Step 1: Import the service and update alignment config parsing**

In `tools/sync_stats.py`, add import:

```python
from src.domain.services.activity_topology_alignment_service import (
    AlignmentGateConfig,
    ActivityTopologyAlignmentService,
)
```

Update `_DEFAULT_ALIGNMENT_GATE`:

```python
_DEFAULT_ALIGNMENT_GATE = {
    "enabled": True,
    "profile": "legacy_exact",
    "min_event_match_ratio": 0.6,
    "min_unique_activity_coverage": 0.6,
    "min_node_coverage": 0.0,
    "on_fail": "write_with_flag",
    "warn_on_fail": True,
    "candidate_node_fields": ["id"],
    "ignore_structural_only_nodes": False,
    "strip_classifier_suffix": False,
    "normalize_case": False,
    "collapse_separators": False,
    "fail_on_ambiguity": True,
}
```

Extend `_resolve_alignment_gate_config(...)` after existing validation:

```python
    profile = _normalize_text(cfg.get("profile", _DEFAULT_ALIGNMENT_GATE["profile"])) or "legacy_exact"
    if profile not in {"legacy_exact", "safe_normalized", "research_strict"}:
        raise ValueError("sync_stats.alignment_gate.profile must be one of {'legacy_exact', 'safe_normalized', 'research_strict'}.")

    base_profile = AlignmentGateConfig.for_profile(profile)
    candidate_node_fields = cfg.get("candidate_node_fields", list(base_profile.candidate_node_fields))
    if isinstance(candidate_node_fields, str):
        candidate_node_fields = [item.strip() for item in candidate_node_fields.split(",") if item.strip()]
    if not isinstance(candidate_node_fields, list) or not candidate_node_fields:
        candidate_node_fields = list(base_profile.candidate_node_fields)
```

Return these additional keys:

```python
        "profile": profile,
        "candidate_node_fields": list(candidate_node_fields),
        "ignore_structural_only_nodes": bool(cfg.get("ignore_structural_only_nodes", base_profile.ignore_structural_only_nodes)),
        "strip_classifier_suffix": bool(cfg.get("strip_classifier_suffix", base_profile.strip_classifier_suffix)),
        "normalize_case": bool(cfg.get("normalize_case", base_profile.normalize_case)),
        "collapse_separators": bool(cfg.get("collapse_separators", base_profile.collapse_separators)),
        "fail_on_ambiguity": bool(cfg.get("fail_on_ambiguity", base_profile.fail_on_ambiguity)),
```

- [x] **Step 2: Replace local summary call with service call**

Replace:

```python
            alignment_summary = _build_alignment_summary(
                events=alignment_events,
                activity_ids=_dto_activity_ids(dto),
                gate_cfg=alignment_gate_cfg,
                scope_used=alignment_scope_used,
            )
```

With:

```python
            alignment_summary = ActivityTopologyAlignmentService().evaluate(
                events=alignment_events,
                dto=dto,
                config=AlignmentGateConfig.from_mapping(alignment_gate_cfg),
                scope_used=alignment_scope_used,
            ).to_dict()
```

Keep `_dto_activity_ids(...)` and `_build_alignment_summary(...)` temporarily until tests pass, then remove them in Task 3 Step 4.

- [x] **Step 3: Run existing alignment tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_sync_stats_tool.py::test_sync_stats_alignment_gate_can_skip_snapshot_on_activity_mismatch tests/application/test_sync_stats_tool.py::test_sync_stats_alignment_gate_write_with_flag_persists_alignment_metadata -v
```

Expected:

```text
2 passed
```

- [x] **Step 4: Remove dead local functions**

After existing tests pass, remove from `tools/sync_stats.py`:

```python
def _dto_activity_ids(...)
def _build_alignment_summary(...)
```

- [x] **Step 5: Run broader sync-stats application tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_sync_stats_tool.py -v
```

Expected:

```text
all tests in tests/application/test_sync_stats_tool.py pass
```

---

## Task 4: Add Real-Object Alignment Gate Tests

**Files:**
- Create: `tests/application/test_sync_stats_alignment_gate_real_objects.py`
- Modify only if needed: `tools/sync_stats.py`

- [x] **Step 1: Create real-object test helper file**

Create `tests/application/test_sync_stats_alignment_gate_real_objects.py` with reusable helpers:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.entities.process_structure import ProcessStructureDTO
from src.infrastructure.repositories.file_based_knowledge_graph_repository import FileBasedKnowledgeGraphRepository
from tools.sync_stats import main as sync_stats_main


def _event(activity: str, ts: str, resource: str = "u1") -> str:
    return f"""
    <event>
      <string key="concept:name" value="{activity}" />
      <string key="lifecycle:transition" value="complete" />
      <date key="time:timestamp" value="{ts}" />
      <string key="org:resource" value="{resource}" />
    </event>"""


def _trace(case_id: str, version: str, activities: list[str]) -> str:
    events = [
        _event(activity, f"2024-01-01T00:{idx + 1:02d}:00+00:00", f"u{idx + 1}")
        for idx, activity in enumerate(activities)
    ]
    return f"""
  <trace>
    <string key="concept:name" value="{case_id}" />
    <string key="concept:version" value="{version}" />
{''.join(events)}
  </trace>"""


def _write_xes(path: Path, activities: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
{_trace("C1", "alpha", activities)}
</log>
""",
        encoding="utf-8",
    )


def _save_structure(kg_dir: Path, nodes: list[dict]) -> FileBasedKnowledgeGraphRepository:
    repo = FileBasedKnowledgeGraphRepository(base_dir=kg_dir)
    node_ids = [str(node["id"]) for node in nodes]
    edges = list(zip(node_ids[:-1], node_ids[1:])) or [(node_ids[0], node_ids[0])]
    repo.save_process_structure(
        "alpha",
        ProcessStructureDTO(version="alpha", allowed_edges=edges, nodes=nodes),
        process_name="alpha",
    )
    return repo


def _write_config(path: Path, *, xes_path: Path, kg_dir: Path, profile: str, on_fail: str, thresholds: tuple[float, float, float]) -> None:
    min_event, min_unique, min_node = thresholds
    path.write_text(
        f"""
data:
  dataset_name: "alpha"
  dataset_label: "alpha"
  log_path: "{xes_path.as_posix()}"

mapping:
  adapter: "xes"
  knowledge_graph:
    backend: "file"
    path: "{kg_dir.as_posix()}"
    strict_load: true
  xes_adapter:
    case_id_key: "concept:name"
    activity_key: "concept:name"
    timestamp_key: "time:timestamp"
    resource_key: "org:resource"
    lifecycle_key: "lifecycle:transition"
    version_key: "concept:version"
    pairing_strategy: "lifo"
    use_classifier: false

sync_stats:
  enabled: true
  stats_time_policy: "strict_asof"
  process_scope_policy: "up_to_target_version"
  windows_days: [7, 30]
  process_filters: ["alpha"]
  alignment_gate:
    enabled: true
    profile: "{profile}"
    min_event_match_ratio: {min_event}
    min_unique_activity_coverage: {min_unique}
    min_node_coverage: {min_node}
    on_fail: "{on_fail}"
    warn_on_fail: false
  show_progress: false

experiment:
  mode: "sync-stats"
  split_strategy: "none"
  train_ratio: 1.0
  fraction: 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _run_sync(cfg_path: Path, out_path: Path) -> dict:
    rc = sync_stats_main(["--config", str(cfg_path), "--out", str(out_path), "--as-of", "2024-01-02T00:00:00Z"])
    assert rc == 0
    return json.loads(out_path.read_text(encoding="utf-8"))
```

- [x] **Step 2: Add real-object test for `Task_Approve` id vs `Approve` name**

Append:

```python
def test_safe_normalized_matches_unique_node_names_in_real_sync_stats(tmp_path: Path):
    xes_path = tmp_path / "alpha.xes"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "summary.json"

    _write_xes(xes_path, ["Approve", "Archive"])
    repo = _save_structure(
        kg_dir,
        [
            {"id": "Task_Approve", "name": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "Task_Archive", "name": "Archive", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
        ],
    )
    _write_config(cfg_path, xes_path=xes_path, kg_dir=kg_dir, profile="safe_normalized", on_fail="raise", thresholds=(1.0, 1.0, 1.0))

    summary = _run_sync(cfg_path, out_path)

    assert summary["processed_versions"] == 1
    loaded = repo.get_process_structure("alpha", process_name="alpha")
    alignment = loaded.metadata["stats_contract"]["alignment"]
    assert alignment["is_aligned"] is True
    assert alignment["event_match_ratio"] == pytest.approx(1.0)
    assert alignment["node_coverage"] == pytest.approx(1.0)
    assert alignment["match_counts_by_strategy"]["exact_name"] == 2
```

- [x] **Step 3: Add real-object test that structural nodes do not depress node coverage**

Append:

```python
def test_safe_normalized_ignores_structural_nodes_in_real_sync_stats(tmp_path: Path):
    xes_path = tmp_path / "alpha.xes"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "summary.json"

    _write_xes(xes_path, ["TaskA", "TaskB"])
    repo = _save_structure(
        kg_dir,
        [
            {"id": "StartEvent", "name": "Start", "bpmn_tag": "startEvent", "type": "startEvent", "activity_type": "startEvent"},
            {"id": "TaskA", "name": "Task A", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "GatewayChoice", "name": "Gateway", "bpmn_tag": "exclusiveGateway", "type": "exclusiveGateway", "activity_type": "exclusiveGateway"},
            {"id": "TaskB", "name": "Task B", "bpmn_tag": "serviceTask", "type": "serviceTask", "activity_type": "serviceTask"},
            {"id": "EndEvent", "name": "End", "bpmn_tag": "endEvent", "type": "endEvent", "activity_type": "endEvent"},
        ],
    )
    _write_config(cfg_path, xes_path=xes_path, kg_dir=kg_dir, profile="safe_normalized", on_fail="raise", thresholds=(1.0, 1.0, 1.0))

    summary = _run_sync(cfg_path, out_path)

    assert summary["processed_versions"] == 1
    loaded = repo.get_process_structure("alpha", process_name="alpha")
    alignment = loaded.metadata["stats_contract"]["alignment"]
    assert alignment["structure_node_count"] == 5
    assert alignment["loggable_node_count"] == 2
    assert alignment["ignored_structural_node_count"] == 3
    assert alignment["node_coverage"] == pytest.approx(1.0)
```

- [x] **Step 4: Add real-object test for ambiguous duplicate name failing closed**

Append:

```python
def test_safe_normalized_ambiguous_name_raises_in_real_sync_stats(tmp_path: Path):
    xes_path = tmp_path / "alpha.xes"
    kg_dir = tmp_path / "kg"
    cfg_path = tmp_path / "cfg.yaml"
    out_path = tmp_path / "summary.json"

    _write_xes(xes_path, ["Approve"])
    _save_structure(
        kg_dir,
        [
            {"id": "TaskA1", "name": "Approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
            {"id": "TaskA2", "name": "approve", "bpmn_tag": "userTask", "type": "userTask", "activity_type": "userTask"},
        ],
    )
    _write_config(cfg_path, xes_path=xes_path, kg_dir=kg_dir, profile="safe_normalized", on_fail="raise", thresholds=(1.0, 1.0, 0.0))

    with pytest.raises(ValueError, match="ambiguous_activity_mapping"):
        sync_stats_main(["--config", str(cfg_path), "--out", str(out_path), "--as-of", "2024-01-02T00:00:00Z"])
```

- [x] **Step 5: Run real-object test file**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/application/test_sync_stats_alignment_gate_real_objects.py -v
```

Expected:

```text
3 passed
```

---

## Task 5: Add BPMN/XES E2E Research Profile Coverage

**Files:**
- Modify: `tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py`

- [x] **Step 1: Change the existing integration config to use `safe_normalized` with strict thresholds**

In `_write_sync_stats_config(...)`, replace:

```yaml
  alignment_gate:
    enabled: true
    min_event_match_ratio: 0.0
    min_unique_activity_coverage: 0.0
    min_node_coverage: 0.0
```

With:

```yaml
  alignment_gate:
    enabled: true
    profile: "safe_normalized"
    min_event_match_ratio: 1.0
    min_unique_activity_coverage: 1.0
    min_node_coverage: 0.75
    on_fail: "raise"
```

Reason: this test has BPMN structural nodes (`StartEvent`, `GatewayChoice`, `EndEvent`) but only loggable tasks in XES. The new denominator should let the test pass without weakening thresholds to zero.

- [x] **Step 2: Assert persisted alignment diagnostics**

After:

```python
    stats_dto = repo.get_process_structure("v1", process_name="loan_proc")
    assert stats_dto is not None
```

Add:

```python
    alignment = stats_dto.metadata["stats_contract"]["alignment"]
    assert alignment["profile"] == "safe_normalized"
    assert alignment["is_aligned"] is True
    assert alignment["event_match_ratio"] == pytest.approx(1.0)
    assert alignment["unique_activity_coverage"] == pytest.approx(1.0)
    assert alignment["loggable_node_count"] == 4
    assert alignment["ignored_structural_node_count"] == 3
```

- [x] **Step 3: Run integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py -v
```

Expected:

```text
1 passed
```

---

## Task 6: Update Config Catalog for New Alignment Policy Keys

**Files:**
- Modify: `configs/ui/config_catalog.yaml`

- [x] **Step 1: Add new optional keys under the existing `sync_stats.alignment_gate.*` block**

Add catalog entries for:

```yaml
  sync_stats.alignment_gate.profile:
    path: sync_stats.alignment_gate.profile
    section: sync_stats
    label: Producer alignment profile
    description: Matching profile for activity-to-topology alignment.
    affects: Alignment denominator, candidate matching strategy, and research strictness.
    default: legacy_exact
    enum:
    - legacy_exact
    - safe_normalized
    - research_strict
    required_in_modes: []
    required_when: {}
    ui:
      tab: eopkg
      group: sync_stats
      priority: 2
      order: 19
```

Also add boolean/list fields:

```yaml
sync_stats.alignment_gate.candidate_node_fields
sync_stats.alignment_gate.ignore_structural_only_nodes
sync_stats.alignment_gate.strip_classifier_suffix
sync_stats.alignment_gate.normalize_case
sync_stats.alignment_gate.collapse_separators
sync_stats.alignment_gate.fail_on_ambiguity
```

Use defaults that match `legacy_exact`:

```text
candidate_node_fields: ["id"]
ignore_structural_only_nodes: false
strip_classifier_suffix: false
normalize_case: false
collapse_separators: false
fail_on_ambiguity: true
```

- [x] **Step 2: Verify YAML syntax**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('configs/ui/config_catalog.yaml', encoding='utf-8')); print('YAML OK')"
```

Expected:

```text
YAML OK
```

---

## Task 7: Add Research Config Example

**Files:**
- Create: `configs/experiments/mvp2_5_stage4_2_research_strict_alignment_example.yaml`

- [x] **Step 1: Create config example based on existing Stage 4.2 stats config**

Create `configs/experiments/mvp2_5_stage4_2_research_strict_alignment_example.yaml`:

```yaml
# Research-grade alignment profile example.
# Copy the relevant data/mapping/model/training blocks from the actual experiment config.

experiment:
  mode: train
  stats_time_policy: strict_asof
  on_missing_asof_snapshot: raise

sync_stats:
  enabled: true
  stats_time_policy: strict_asof
  process_scope_policy: up_to_target_version
  alignment_gate:
    enabled: true
    profile: research_strict
    min_event_match_ratio: 0.9
    min_unique_activity_coverage: 0.9
    min_node_coverage: 0.8
    on_fail: raise
    warn_on_fail: true
```

- [x] **Step 2: Decide whether this should be a runnable config or a snippet**

If the project convention rejects partial example configs, place the snippet in `docs/runbooks/mvp2_5-commands.md` instead and do not create this YAML file. Prefer not to add non-runnable configs to `configs/experiments/` unless the repo already has examples of that style.

---

## Task 8: Documentation Updates

**Files:**
- Modify: `docs/LLD_MVP2_5.MD`
- Modify: `docs/DATA_MODEL_MVP2_5.MD`
- Modify: `docs/runbooks/mvp2_5-commands.md`
- Modify: `docs/current/architecture-debt.md`
- Modify: `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`
- Modify only if routing changes: `AGENTS.MD`

- [x] **Step 1: Update `docs/LLD_MVP2_5.MD`**

Document:

```md
### Producer Alignment Profiles

- `legacy_exact`: exact `activity_def_id` to node id matching; compatibility mode.
- `safe_normalized`: loggable-node denominator; id/name candidates; case/separator normalization; classifier suffix stripping; ambiguity fails closed.
- `research_strict`: safe matching semantics with locked high thresholds and fail-fast policy.
```

- [x] **Step 2: Update `docs/DATA_MODEL_MVP2_5.MD`**

Extend alignment block fields:

```md
Alignment block includes:
1. `profile`
2. `is_aligned`
3. `alignment_reason`
4. `alignment_failures`
5. `event_match_ratio`
6. `unique_activity_coverage`
7. `node_coverage`
8. `structure_node_count`
9. `loggable_node_count`
10. `ignored_structural_node_count`
11. `matched_event_count`
12. `matched_unique_activity_count`
13. `matched_loggable_node_count`
14. `unmatched_event_activities_top`
15. `ambiguous_event_activities_top`
16. `match_counts_by_strategy`
17. `candidate_node_fields`
```

- [x] **Step 3: Update `docs/runbooks/mvp2_5-commands.md`**

Add research-grade snippet:

```yaml
experiment:
  stats_time_policy: strict_asof
  on_missing_asof_snapshot: raise

sync_stats:
  alignment_gate:
    enabled: true
    profile: research_strict
    min_event_match_ratio: 0.9
    min_unique_activity_coverage: 0.9
    min_node_coverage: 0.8
    on_fail: raise
```

- [x] **Step 4: Update debt and ADR**

If all tests pass:

In `docs/current/architecture-debt.md`, change `activity_to_topology_alignment_gate`:

```md
- `status`: closed
- `current_behavior`: producer-side alignment profiles and strict research profile are implemented
- `target_state`: closed; maintain catalog/docs sync for future config keys
```

In `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`, change:

```md
Status: Accepted
```

Add runtime rules:

```md
1. Research-grade runs use `sync_stats.alignment_gate.profile: research_strict`.
2. Ambiguous mappings fail closed.
3. `node_coverage` denominator uses loggable topology nodes, not all BPMN nodes.
```

---

## Task 9: Validation Pack

**Files:**
- No edits unless validation exposes failures.

- [x] **Step 1: Run architecture guard**

Run:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Expected:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

- [x] **Step 2: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_activity_topology_alignment_service.py tests/application/test_sync_stats_alignment_gate_real_objects.py tests/application/test_sync_stats_tool.py tests/integration/test_bpmn_xes_stats_to_gnn_cycle.py -v
```

Expected:

```text
all selected tests pass
```

- [x] **Step 3: Run MVP1 regression gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Expected:

```text
all selected mvp1_regression tests pass
```

- [x] **Step 4: Run full tests if time allows**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected:

```text
all tests pass
```

If full tests are too slow, report exactly which focused tests were run and why full suite was not run.

---

## Acceptance Criteria

The `activity_to_topology_alignment_gate` debt is closed only when:

1. Alignment computation is in `src/domain/services/activity_topology_alignment_service.py`.
2. `tools/sync_stats.py` delegates alignment computation to the domain service.
3. `legacy_exact` preserves current behavior.
4. `safe_normalized` matches unique node names and normalized id/name variants.
5. Structural-only BPMN nodes are excluded from `node_coverage` denominator in safe/research profiles.
6. Ambiguous mappings fail closed with persisted diagnostics.
7. `write_with_flag`, `skip_snapshot`, and `raise` remain supported.
8. Real-object tests cover name shift, structural denominator, and ambiguity.
9. BPMN/XES integration test uses non-zero strict alignment thresholds.
10. Config catalog documents every new `sync_stats.alignment_gate.*` key.
11. ADR-0006 is accepted.
12. `docs/current/architecture-debt.md` marks this debt closed.
13. Documentation drift check is included in final response.

---

## Implementation Notes

- Do not switch the global default profile from `legacy_exact` to `safe_normalized` in the same first implementation unless all existing sync-stats tests and the integration test pass without lowering thresholds.
- If default remains `legacy_exact`, the P0 debt can still close if `research_strict` is implemented, documented, and tested.
- Do not use fuzzy matching.
- Do not hardcode global `Task_` prefix stripping.
- Keep all matching deterministic and explainable through `match_counts_by_strategy`.
