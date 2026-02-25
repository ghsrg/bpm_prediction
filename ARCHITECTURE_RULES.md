# ARCHITECTURE_RULES.md

## 1) Purpose
This document makes Clean Architecture + Hexagonal rules **enforceable** for `bpm_prediction`.

---

## 2) Layer responsibilities
- **Domain (`src/core/`)**: pure business/scientific logic, entities, policies, interfaces.
- **Application (`src/pipeline/`)**: use-case orchestration and workflow coordination.
- **Adapters (`src/adapters/`)**: infrastructure implementations for DB, API, tracking, reporting.
- **Frameworks & Drivers**: external systems and runtimes.

---

## 3) Allowed dependency directions
- `src/core/` → standard libs, math/ML libs, internal core modules only.
- `src/pipeline/` → `src/core/interfaces/` (ports) only.
- `src/adapters/` → infrastructure libraries + `src/core/interfaces/`.
- Dependency inversion is mandatory:
  - Application depends on Ports.
  - Adapters implement Ports.

Forbidden:
- Domain importing adapters/pipeline.
- Pipeline importing concrete adapters.
- Adapters importing concrete pipeline logic.

---

## 4) Ports rules
- All ports live in `src/core/interfaces/`.
- All outbound interactions from Application must go through those ports.
- Concrete adapters must inherit from corresponding interface classes.

---

## 5) Infrastructure leakage policy
Domain layer must not import infrastructure technologies (examples):
- `mlflow`
- `neo4j`
- `sqlalchemy`
- `fastapi`
- concrete adapter/pipeline modules

No direct infrastructure types are allowed inside Domain contracts.

---

## 6) Validation mechanism
Architecture constraints are validated by:
- `tools/architecture_guard.py`
- `make arch-check`
- CI workflow: `.github/workflows/architecture-guard.yml`
- local Git hook: `.git/hooks/pre-commit`

Merge policy: architecture violations block commit/CI.
