# ETALON_TRAINING_FLOW

## 1. Training flow (process prediction)
1. Завантаження `global_statistics` (`node_count.max`, `edge_count.max`).
2. `prepare_data(...)` або `prepare_data_log_only(...)` -> список `Data`, `input_dim`, `doc_dim`, `global_node_dict`.
3. Split (`train/val/test`).
4. На кожній епосі:
   - `train_epoch(...)` повертає тільки `avg_loss`.
   - `calculate_statistics(...)` повертає dict із quality метриками + prefix stats.
   - scheduler/early stopping/checkpointing.

## 2. Що реально повертається з train_step
У etalon **немає окремого `train_step` API**.
Функція, яка виконує кроки train, — `train_epoch`, її return = `float avg_loss`.

## 3. Metrics, які реально логуються
### 3.1 Process prediction (`*_pr`)
- `accuracy`
- `top_k_accuracy`
- `precision`
- `recall`
- `f1_score`
- `mae`, `rmse`, `r2` (для time head)
- `out_of_scope_rate`
- `confusion_matrix`
- `activity_train_vs_val_accuracy`
- `prefix_stats` (агрегація по довжині префікса, включно з `top1/top3/top5`, confidence, out_of_scope)

### 3.2 Anomaly path (`core_gnn.py`)
- `precision`, `recall`, `roc_auc`, `f1_score`, `auprc`
- `adr`, `far`, `fpr`, `fnr`
- `confusion_matrix`

## 4. Metadata, які реально передаються
- Run-level metadata (`experiment_id`, `git_commit`) в pipeline не передаються.
- Stage-level metadata (`dataset_id`, `schema_version`, `process_version`, ...) в tensor/train API не передаються.
- Натомість в `Data` і train-контексті є:
  - `doc_features` (вектор з `doc_info`),
  - `node_ids` (для out-of-scope перевірок),
  - `global_node_dict` (поза `Data`, як аргумент функцій).

## 5. OOD / drift
- Окремого drift detector (Wasserstein/концепт дріфт) у тренувальному контурі не знайдено.
- Реалізовано тільки метрику `out_of_scope_rate` у process prediction:
  - передбачення/label відображають у `global_node_dict`;
  - якщо індекс невалідний для конкретного графа (`node_ids`), приклад вважається out-of-scope.

## 6. Неявні інваріанти
- Для prefix-сценарію потрібні валідні `SEQUENCE_COUNTER_ > 0` щонайменше для 2 вузлів.
- `global_statistics` має існувати і містити `node_count.max`, `edge_count.max`.
- Всі `doc_features` мають фіксовану довжину між елементами батча.
- `edge_index.size(1) == edge_attr.size(0)` (перевіряється явно, інакше помилка).
