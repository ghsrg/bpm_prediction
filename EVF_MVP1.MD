# EVF_MVP1.MD (Evaluation Framework)

## 1. Статус
`APPROVED` (Scope: MVP1 - Baseline Evaluation)

## 2. Призначення
Специфікація методології оцінювання для MVP1 baseline-моделі з фіксованими анти-лікейдж інваріантами.

## 3. Стратегія поділу даних (Data Splitting)
Використовується лише **Strict Temporal Split** на рівні трас.

### 3.1. Temporal Split
1. Траси `RawTrace` сортуються за часом першої події.
2. Розбиття: Train / Validation / Test у заданих пропорціях.
3. Усі префікси однієї траси належать одному спліту.

---

## 4. Метрики (Global Test)
### 4.1. Epoch Tracking
- `train_loss`, `val_loss`
- `train_macro_f1`, `val_macro_f1`, `val_weighted_f1`
- `epoch_duration_sec`

### 4.2. Final Test Metrics
- `test_macro_f1`
- `test_weighted_f1`
- `test_accuracy`
- `test_top3_accuracy`
- `test_precision_macro`
- `test_recall_macro`
- `test_ece`
- `test_inference_time_ms_per_graph`

---

## 5. Drift Evaluation Protocol

### 5.1. Anti-Leakage Invariant
Для `eval_drift` обов'язковий порядок кроків:
\[
\text{chronological sort} \rightarrow \text{window generation} \rightarrow \text{evaluation} \rightarrow \text{logging}
\]

Формально, для трас \(\tau_i\):
\[
\tau_i \preceq \tau_j \iff t_{start}(\tau_i) \le t_{start}(\tau_j),\; i<j
\]
Генерація вікон:
\[
W_k = \mathcal{T}[s_k:s_k+w), \quad s_{k+1}=s_k+\Delta
\]
де \(w\) — `window_size`, \(\Delta\) — `window_stride` (або \(\Delta=w-overlap\)).

### 5.2. Per-window logging (required)
Для кожного вікна логуються:
- `window_start_ts`
- `window_end_ts`
- `window_macro_f1`
- `window_test_ece`

Рекомендовано також логувати ідентифікатори діапазону:
- `window_index`
- `window_start_trace`
- `window_end_trace`

---

## 6. Experiment Tracking (MLflow)
Кожен запуск `train/eval_cross_dataset/eval_drift` логується через `ITracker`.

Обов'язково:
- параметри моделі/даних;
- метрики епох;
- фінальні тестові метрики;
- per-window drift метрики (для `eval_drift`).
