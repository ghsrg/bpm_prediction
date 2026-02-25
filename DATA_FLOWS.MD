# DATA_FLOWS.MD

## Статус
`IN_PROGRESS (DOC-007)`

## 1) Призначення
Документ формалізує потоки даних для DOC-007 за правилом `Input/Output/Metadata` між стадіями pipeline.

---

## 2) End-to-end sequence (runtime)
1. API/CLI отримує `process_instance_id` + `source_type`.
2. Reader adapter повертає `EventLogBatch`.
3. Data Converter нормалізує batch у canonical schema.
4. Graph Builder формує `ProcessGraphDTO` (IG + EPOKG context).
5. Tensor Adapter формує `TensorGraphDTO`.
6. Core Predictor повертає прогноз і confidence.
7. Drift Monitor обчислює drift/OOD сигнали.
8. Reliability Semaphore встановлює `semaphore_mode`.
9. Tracker/Report зберігає метрики та артефакти.
10. Контур повертає `InferenceResultDTO`.

---

## 3) Stage contracts

### 3.1 Ingestion
- **Input:** `process_instance_id`, `source_type`, optional filters.
- **Output:** `EventLogBatch`.
- **Metadata:** `dataset_id`, `schema_version`, `process_version`.

### 3.2 Graph Builder
- **Input:** `EventLogBatch`.
- **Output:** `ProcessGraphDTO`.
- **Metadata:** + `graph_build_version`, `epokg_snapshot_id`.

### 3.3 Tensor Adapter
- **Input:** `ProcessGraphDTO`.
- **Output:** `TensorGraphDTO`.
- **Metadata:** + `feature_config_version`.

### 3.4 Core ML
- **Input:** `TensorGraphDTO`.
- **Output:** `logits/prediction + confidence`.
- **Metadata:** + `model_version`, `checkpoint_id`.

### 3.5 Evaluation
- **Input:** predictions + labels (де доступно).
- **Output:** metrics (`Accuracy/F1/OOS`).
- **Metadata:** + `evaluation_profile`.

### 3.6 Drift Monitor
- **Input:** batch statistics, embeddings, distribution features.
- **Output:** `data_drift_score`, `concept_drift_score`, `wasserstein_drift`, `ood_flag`.
- **Metadata:** + `drift_strategy_version`.

### 3.7 Inference / Delivery
- **Input:** prediction + drift signals.
- **Output:** `InferenceResultDTO`.
- **Metadata:** + `reliability_score`, `semaphore_mode`, `routing_decision`.

---

## 4) Failure routing (minimal)
- `green`: стандартна доставка прогнозу.
- `yellow`: доставка з попередженням + постановка у чергу адаптації.
- `red`: блок автодії + маршрутизація Human-in-the-Loop.

---

## 5) Invariants
1. Без `process_version (κ)` пакет не проходить далі Graph Builder.
2. Без `schema_version` дані вважаються невалідними.
3. Всі stage повертають структуроване `metadata`.
