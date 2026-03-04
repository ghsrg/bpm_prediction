# EVF_MVP1.MD (Evaluation Framework)

## 1. Статус
`APPROVED` (Scope: MVP1 - Baseline Evaluation)

## 2. Призначення
Специфікація методології тестування, метрик та стратегії поділу даних для оцінки ефективності базової нейромережі (Baseline GNN). Формує наукову точку відліку для порівняння з цільовою методологією (Domain-Conditioned GNN) у наступних фазах.

## 3. Стратегія поділу даних (Data Splitting)

Категорично **заборонено** використовувати випадковий поділ (Random Split) або K-Fold Cross Validation на перемішаних трасах. Використовується виключно **Strict Temporal Split** (хронологічний поділ).

### 3.1. Алгоритм Temporal Split
1. Усі траси (`RawTrace`) сортуються за часом першої події (`time_since_case_start = 0.0`).
2. Датасет розбивається у пропорції:
   - **Train:** 70% (найстаріші траси).
   - **Validation:** 10% (для early stopping та налаштування гіперпараметрів).
   - **Test:** 20% (найновіші траси, які модель ніколи не бачила).
3. **Захист від витоку даних (Data Leakage):** Усі префікси однієї траси повинні потрапити суворо в один спліт. Розривати трасу між Train і Test заборонено.

## 4. Цільові метрики (Metrics & Artifacts)

Оскільки бізнес-логи мають сильний дисбаланс класів, `Accuracy` розглядається лише як допоміжна метрика. Регресійні метрики (MAE, MSE) та Out-of-Scope (OOS) Rate у MVP1 відключені (фокус виключно на класифікації без наявності нормативного EOPKG).

### 4.1. Метрики епохи (Online Tracking)
Логуються на кожній епосі під час Train / Validation:
- `loss`: Cross-Entropy.
- `macro_f1`: Основна метрика збалансованості.
- `epoch_duration_sec`: Час проходження епохи.

### 4.2. Фінальні метрики тестування (Test Evaluation)
Розраховуються один раз на тестовому спліті (на базі `sklearn.metrics`):
- `test_macro_f1` (Primary)
- `test_accuracy` (Secondary)
- `test_top3_accuracy`
- `test_precision_macro`
- `test_recall_macro`
- `test_ece`: Expected Calibration Error (якість імовірнісного прогнозу для майбутнього семафора надійності).
- `inference_latency_ms`: Середній час прямого поширення ($f_\theta$) для одного графа екземпляра (фіксує здатність до real-time роботи).

### 4.3. Артефакти (Artifacts)
Зберігаються як фізичні файли після завершення прогону:
- `confusion_matrix.png` (візуалізація матриці помилок).
- `classification_report.txt` (повний звіт `sklearn` по кожному класу активності).
- `model_weights.pt` (збережені ваги нейромережі).

## 5. Experiment Tracking (MLflow)

Кожен запуск `Trainer` або `Evaluator` обов'язково логується через `ITracker` (MLflow).

**Обов'язкові параметри (Parameters) для логування:**
- `model_type`: "Baseline_GNN_MVP1"
- `dataset_id`: (наприклад, "bpic_2012")
- `num_layers`: кількість шарів GNN
- `hidden_dim`: розмірність $d_V$
- `learning_rate`
- `batch_size`

**Обов'язкові метрики (Metrics) для логування:**
- Усі метрики з п. 4.1 (по епохах).
- Усі метрики з п. 4.2 (фінальні результати).
- Збереження артефактів з п. 4.3.