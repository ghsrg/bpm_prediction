# ARCHITECTURE_RULES.MD

## 1) Purpose
This document makes Clean Architecture + Hexagonal rules **enforceable** for `bpm_prediction`.

---

## 2) Layer responsibilities
- **Domain (`src/domain/`)**: pure business/scientific logic, entities (DTOs like `RawTrace`, `EventRecord`), and domain services (Feature Engineering, Prefix Policy).
- **Application (`src/application/`)**: use-case orchestration (Trainer, Evaluator) and interface definitions (Ports).
- **Adapters (`src/adapters/`)**: infrastructure implementations for DB, API, tracking, reporting (e.g., `XESAdapter`, `MLflowTracker`).
- **Frameworks & Drivers**: external systems and runtimes.

---

## 3) Allowed dependency directions
- `src/domain/` в†’ standard libs, math/ML libs, internal domain modules only. MUST NOT import `application` or `adapters`.
- `src/application/` в†’ depends on `src/domain/` (entities/services) and standard libs. MUST NOT import `adapters`.
- `src/adapters/` в†’ infrastructure libraries + `src/application/ports/` + `src/domain/entities/`.
- Dependency inversion is mandatory:
  - Application defines Ports (`src/application/ports/`).
  - Adapters implement those Ports.

Forbidden:
- Domain importing adapters/application.
- Application importing concrete adapters.
- Adapters importing concrete application use-cases.

---

## 4) Ports rules
- All ports live in `src/application/ports/`.
- Defined using Python's `typing.Protocol` (e.g., `IXESAdapter`).
- All outbound interactions from Application must go through those ports.
- Concrete adapters (`src/adapters/`) must implement these protocol interfaces implicitly or explicitly.

---

## 5) Infrastructure leakage policy
Domain layer must not import infrastructure technologies (examples):
- `mlflow` (tracking calls must go through `ITracker` port)
- `neo4j`
- `sqlalchemy`
- `fastapi`
- concrete adapter/pipeline modules

No direct infrastructure types are allowed inside Domain contracts.

**Exceptions (Cross-Cutting Concerns):**
Standard libraries (`datetime`, `typing`, `math`), ML frameworks (`torch`, `torch_geometric`), and global cross-cutting utilities (`loguru`) are **ALLOWED** across all layers.---

---

## 6) Validation mechanism
Architecture constraints are validated by:
- `tools/architecture_guard.py`
- `make arch-check`
- CI workflow: `.github/workflows/architecture-guard.yml`
- local Git hook: `.git/hooks/pre-commit`

Merge policy: architecture violations block commit/CI.

