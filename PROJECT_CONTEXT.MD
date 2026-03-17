# PROJECT_CONTEXT.md

Updated: 2026-03-17  
Purpose: short operational context for agents during MVP2.5 Stage 3.2 -> Stage 4 transition.

---

## 1. Current Project State

`bpm_prediction` is running in a decoupled architecture:

- Offline topology ingestion is active (`main.py ingest-topology`).
- Training and evaluation consume topology from `IKnowledgeGraphPort`.
- Repository backends currently implemented: `in_memory`, `file`.
- Camunda Stage 3.1 runtime adapters are active (cleanup-aware, fallback-safe).
- Camunda Stage 3.2 BPMN structure ingestion is implemented.

MVP1 baseline remains mandatory and protected by regression tests.

---

## 2. MVP1 Baseline Contract (Must Not Break)

Stable data path:

`XESAdapter -> RawTrace -> PrefixPolicy -> PrefixSlice -> BaselineGraphBuilder -> GraphTensorContract -> ModelTrainer`

Invariant behavior:

- `SchemaResolver.resolve_value` is identity in MVP1.
- Categorical unknown maps to `<UNK>=0`.
- Missing numeric values use zero-imputation in normalized space.
- No critic/semaphore/beta dynamic regularization in MVP1 runtime path.

Mandatory checks before merge:

1. `py -3 tools/architecture_guard.py`
2. `pytest -m mvp1_regression -v`
3. `pytest tests/ -v`

---

## 3. MVP2.5 Snapshot (Actual Implementation)

Implemented:

- `FileBasedKnowledgeGraphRepository` with atomic JSON write.
- `InMemoryNetworkXRepository` for tests and lightweight flows.
- `ingest-topology` command for offline structure build.
- `visualize-topology` default mode from repository (`--from-raw` for legacy quick check).
- `DynamicGraphBuilder` structure read by process/version with graceful fallback.
- Camunda BPMN parser path with subprocess flattening and `call_bindings`.

Behavioral rules:

- Training must not trigger synchronous topology extraction.
- Missing topology does not hard-fail unless `mapping.knowledge_graph.strict_load=true`.
- `structure_from_logs` is explicit fallback for Camunda when BPMN source is configured.

---

## 4. Config Ownership Policy

Dataset/source facts belong to `data` and `mapping`.
Experiment knobs belong to `experiment`.

Experiment-level controls include:

- `mode`
- `split_strategy`
- `train_ratio`
- `fraction`
- `split_ratio`
- drift window parameters (`drift_window_size`, `drift_window_sliding`)

Rule: do not duplicate data configs per experiment; vary `experiment` and model config includes.

---

## 5. MVP2.5 Operational Commands

1. Offline structure ingestion:
```bash
python main.py ingest-topology --config <experiment.yaml> --out outputs/<summary>.json
```

2. Train:
```bash
python main.py --config <experiment.yaml>
```

3. Drift evaluation:
```bash
python main.py --config <eval_drift_experiment.yaml>
```

4. Topology visualization from repository:
```bash
python main.py visualize-topology --config <experiment.yaml> --version <version_key> --out outputs/<file>.png
```

---

## 6. Next Architectural Focus

Current priority is MVP2.5 Stage 4:

- switch knowledge backend from file to Neo4j under the same `IKnowledgeGraphPort`,
- keep ingestion/train contract unchanged,
- preserve MVP1 and MVP2.5 non-regression guarantees.

BPMS enrichment remains optional and should be layered after Neo4j parity is stable.

---

## 7. Priority of Truth

1. `AGENT_GUIDE.MD`
2. `PROJECT_CONTEXT.md`
3. `docs/ARCHITECTURE_GUIDELINES.MD`
4. `docs/ARCHITECTURE_RULES.MD`
5. `docs/TARGET_ARCHITECTURE.MD`
6. Phase docs (`ARCHITECTURE_MVPx`, `DATA_MODEL_MVPx`, `DATA_FLOWS_MVPx`, `LLD_MVPx`, `EVF_MVPx`)
