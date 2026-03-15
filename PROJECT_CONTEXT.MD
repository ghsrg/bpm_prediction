# PROJECT_CONTEXT.md

Updated: 2026-03-11  
Purpose: short operational context for agents during MVP1->MVP2 transition.

---

## 1. Current Project State

`bpm_prediction` runs in dual-track mode:

- MVP1 is stable and treated as baseline runtime.
- MVP2 is the active implementation track (EOPKG, structural constraints, fusion model).

Hard requirement: MVP1 experiments must stay runnable at any time.

---

## 2. MVP1 Baseline Contract (Must Not Break)

Stable data path:

`XESAdapter -> RawTrace -> PrefixPolicy -> PrefixSlice -> BaselineGraphBuilder -> GraphTensorContract -> ModelTrainer`

Invariant behavior:

- `SchemaResolver.resolve_value` is identity in MVP1.
- Categorical unknown maps to `<UNK>=0`.
- Missing numeric values use zero-imputation in normalized space.
- No EOPKG, structural mask gating, critic, reliability semaphore, or continual-learning logic in MVP1 runtime path.

Mandatory pre-merge checks during transition:

1. `py -3 tools/architecture_guard.py`
2. `pytest tests/ -v`
3. At least one MVP1 smoke run via `python main.py --config <mvp1_config.yaml>`

---

## 3. MVP2 Start Point (Reality Check)

Documentation for MVP2 exists and is detailed (`ARCHITECTURE_MVP2`, `DATA_MODEL_MVP2`, `DATA_FLOWS_MVP2`, `LLD_MVP2`, `EVF_MVP2`), but codebase is still mostly MVP1 runtime:

- Available models: `BaselineGCN`, `BaselineGATv2`.
- No implemented `EOPKGFusionGNN`.
- No implemented `TopologyExtractorService` runtime integration.
- `GraphTensorContract` runtime flow still uses baseline fields only.

This means MVP2 should start as incremental implementation with strict backward compatibility gates.

---

## 4. Config Ownership Policy

Dataset facts belong to `data`.
Experiment knobs belong to `experiment`.

Experiment-level controls include:

- `mode`
- `split_strategy`
- `train_ratio`
- `fraction`
- `split_ratio`
- drift window parameters (`drift_window_size`, sliding/stride/overlap policy)

Rule: do not duplicate dataset configs for each experiment; vary only `experiment` blocks.

---

## 5. MVP1 Experiments That Must Stay Handy

- `configs/experiments/01_train_bpi2012.yaml`
- `configs/experiments/01_eval_drift_bpi2012.yaml`
- `configs/experiments/01_eval_cross_dataset_bpi2012.yaml`

Baseline run command:

```bash
python main.py --config configs/experiments/01_train_bpi2012.yaml
```

---

## 6. MVP2 Implementation Policy

When adding MVP2:

- Add new code paths behind explicit mode/feature flags.
- Keep defaults equivalent to MVP1 behavior.
- Preserve existing MVP1 config compatibility.
- Add focused tests per change (unit + integration where orchestration changes).
- Do not rewrite baseline components unless required by extension boundaries.

---

## 7. Priority of Truth

1. `AGENT_GUIDE.MD`
2. `PROJECT_CONTEXT.md`
3. `docs/ARCHITECTURE_GUIDELINES.MD`
4. `docs/ARCHITECTURE_RULES.MD`
5. MVP docs (`ARCHITECTURE_MVP1/2`, `DATA_MODEL_MVP1/2`, `DATA_FLOWS_MVP1/2`, `LLD_MVP1/2`, `EVF_MVP1/2`)


---

## 8. MVP2.5 Snapshot (2026-03-15)

Current baseline:
- Offline topology ingestion is implemented (`main.py ingest-topology`).
- Knowledge repository backends: `in_memory` and `file`.
- Training and eval consume topology through repository interface.
- Camunda runtime Stage 3.1 includes cleanup-aware fallback diagnostics.

New canonical docs:
- `docs/ARCHITECTURE_MVP2_5.MD`
- `docs/DATA_FLOWS_MVP2_5.MD`
- `docs/DATA_MODEL_MVP2_5.MD`
- `docs/LLD_MVP2_5.MD`
- `docs/EVF_MVP2_5.MD`

Before Stage 3.2 BPMN parser work:
- keep MVP1 regression shield green,
- keep ingestion and train pipelines decoupled,
- keep repository contract stable for future Neo4j backend.
