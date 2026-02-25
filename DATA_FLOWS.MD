# DATA_FLOWS.MD

## Статус
`IN_PROGRESS (DOC-007)`

## 1) Призначення
Документ формалізує потоки даних для DOC-007 за правилом `Input/Output/Metadata` між стадіями pipeline.

---

## 2) Canonical metadata policy

### 2.1 Run-level metadata (для всього запуску)
Задається на старті запуску, проходить через усі стадії:
- `experiment_id`
- `git_commit`

### 2.2 Stage-level required (для кожного stage)
- `dataset_id`
- `schema_version`
- `process_version` (`κ`)

### 2.3 Stage-level optional/conditional
- `model_version`, `checkpoint_id` — для `Core ML`, `Evaluation`, `Inference`.
- `epokg_snapshot_id`, `graph_build_version` — для `Graph Builder`.
- `feature_config_version` — для `Tensor Adapter`.
- `drift_strategy_version` — для `Drift Monitor`.
- `evaluation_profile` — для `Evaluation`.
- `routing_decision` — для `Inference`.

---

## 3) End-to-end sequence (runtime)
1. API/CLI отримує `process_instance_id` + `source_type` і ініціалізує `RunMetadataDTO`.
2. Reader adapter (`IDataReader`/`IXESReader`) повертає `EventLogBatch`.
3. `IDataConverter` нормалізує batch у canonical schema.
4. `IGraphBuilder` формує `ProcessGraphDTO` (IG + EPOKG context).
5. `ITensorAdapter` формує `TensorGraphDTO`.
6. `ICorePredictor.infer` повертає первинний прогноз і confidence.
7. `IDriftMonitor.evaluate_drift` обчислює drift/OOD сигнали з `DriftInputDTO`.
8. `IReliabilitySemaphore.compute` обчислює `reliability_score` і встановлює `semaphore_mode`.
9. `ITracker`/`IReportWriter` зберігають метрики та артефакти.
10. Контур повертає `InferenceResultDTO`.

---

## 4) Stage contracts

### 4.1 Ingestion
- **Ports:** `IDataReader`, `IXESReader`.
- **Input:** `process_instance_id`, `source_type`, optional filters, `RunMetadataDTO`.
- **Output:** `EventLogBatch`.
- **Metadata:**
  - Run-level: `experiment_id`, `git_commit`.
  - Stage-level required: `dataset_id`, `schema_version`, `process_version`.

### 4.2 Graph Builder
- **Ports:** `IGraphBuilder`.
- **Input:** `EventLogBatch`.
- **Output:** `ProcessGraphDTO`.
- **Metadata:**
  - Run-level: inherited.
  - Stage-level required: inherited.
  - Optional: `graph_build_version`, `epokg_snapshot_id`.

### 4.3 Tensor Adapter
- **Ports:** `ITensorAdapter`.
- **Input:** `ProcessGraphDTO`.
- **Output:** `TensorGraphDTO`.
- **Metadata:**
  - Run-level: inherited.
  - Stage-level required: inherited.
  - Optional: `feature_config_version`.

### 4.4 Core ML
- **Ports:** `ICorePredictor`.
- **Input:** `TensorGraphDTO`.
- **Output (infer):** `InferenceResultDTO`.
- **Output (train_step):** `TrainStepResultDTO` з ключами `loss_total`, `loss_task`, `loss_reg`, `lr`, `grad_norm`.
- **Metadata:**
  - Run-level: inherited.
  - Stage-level required: inherited.
  - Optional/required-by-stage: `model_version`, `checkpoint_id`.

### 4.5 Evaluation
- **Ports:** `ITracker`, `IReportWriter`.
- **Input:** predictions + labels (де доступно).
- **Output:** metrics (`Accuracy/F1/OOS`).
- **Metadata:**
  - Run-level: inherited.
  - Stage-level required: inherited.
  - Optional/required-by-stage: `model_version`, `checkpoint_id`, `evaluation_profile`.

### 4.6 Drift Monitor
- **Ports:** `IDriftMonitor`.
- **Input:** `DriftInputDTO` (`EventLogBatch`, optional `TensorGraphDTO`, distribution features, embedding summary, counters).
- **Output:** `DriftResultDTO`:
  - `data_drift_score`
  - `concept_drift_score`
  - `wasserstein_drift`
  - `ood_flag`
- **Metadata:**
  - Run-level: inherited.
  - Stage-level required: inherited.
  - Optional: `drift_strategy_version`.

### 4.7 Reliability / Inference delivery
- **Ports:** `IReliabilitySemaphore`.
- **Input:** первинний prediction + `DriftResultDTO`.
- **Output:** фінальний `InferenceResultDTO` з `reliability_score`, `semaphore_mode`, `ood_flag`.
- **Metadata:**
  - Run-level: inherited.
  - Stage-level required: inherited.
  - Optional: `model_version`, `checkpoint_id`, `routing_decision`.

---

## 5) Failure routing (minimal)
- `green`: стандартна доставка прогнозу.
- `yellow`: доставка з попередженням + постановка у чергу адаптації.
- `red`: блок автодії + маршрутизація Human-in-the-Loop.

---

## 6) Invariants
1. Без `process_version (κ)` пакет не проходить далі Graph Builder.
2. Без `schema_version` дані вважаються невалідними.
3. Всі stage повертають структуровані `run_metadata` та `stage_metadata`.
4. `IDriftMonitor` не може працювати лише з metadata — потрібен `DriftInputDTO` з фактичними ознаками розподілу.
5. `IReliabilitySemaphore` є обов'язковим окремим модулем; його не можна підміняти неявною логікою в іншому stage.

---

## 7) Validation Tests (Mandatory)

### Test A — End-to-End DTO Chain
Перевірити безперервність DTO:
`EventLogBatch → ProcessGraphDTO → TensorGraphDTO → InferenceResultDTO`.
Не допускаються пропуски required-ідентифікаторів (`case_id`, `process_version (κ)`, `dataset_id`, `schema_version`).

### Test B — Metadata Consistency
Правила metadata мають бути ідентичними в `DATA_FLOWS.MD` і `INTERFACE_CONTRACTS.MD`.
Run-level vs Stage-level вимоги не можуть конфліктувати.

### Test C — Module Coverage
Кожен модуль, згаданий у flow, повинен мати відповідний Port/Interface у `INTERFACE_CONTRACTS.MD`.
Phantom-модулі заборонені.

### Test D — Contract Concreteness
Для кожного DTO мають бути визначені мінімальні інваріанти та required-поля.
Мінімум для `ProcessGraphDTO`: required-ключі вузлів/ребер і allowed `edge_type`.

**Definition of Done:** заборонено merge/PR, якщо будь-який тест A–D не проходить.
