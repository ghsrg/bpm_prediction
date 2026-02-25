# ETALON_TENSOR_SCHEMA

## 1. Загальний факт
Etalon використовує `torch_geometric.data.Data`, але має щонайменше дві різні фактичні схеми:
1. **Anomaly detection path** (`core_gnn.py`).
2. **Process prediction path** (`core_*_pr.py`, напр. `core_GATConv_pr.py`).

---

## 2. Anomaly detection (`core_gnn.py`)
### 2.1 Поля `Data`
- `x: FloatTensor [N, F_node]`
- `edge_index: LongTensor [2, E]`
- `edge_attr: FloatTensor [E, F_edge]`
- `y: FloatTensor [1]` де `0=normal`, `1=anomalous`
- `doc_features: FloatTensor [F_doc]`

### 2.2 F_node / F_edge / F_doc (фактичний відбір)
- Node attrs: `type`, `DURATION_`, `START_TIME_`, `END_TIME_`, `active_executions`, `SEQUENCE_COUNTER_`, `overdue_work`, `duration_work`.
- Edge attrs: `DURATION_E`, `taskaction_code_E`, `overdue_work_E` (на практиці третій часто нульовий, бо граф має `duration_work_E`).
- Doc attrs: `PurchasingBudget`, `InitialPrice`, `FinalPrice`, `ExpectedDate`, `CategoryL1`, `CategoryL2`, `CategoryL3`, `ClassSSD`, `Company_SO`.

### 2.3 Формула label `y`
- `y = 0` для графів з реєстру `normal_graphs`.
- `y = 1` для графів з `anomalous_graphs`, відфільтрованих за `params.str.contains(anomaly_type)`.

### 2.4 Маски/батчинг
- Спеціальні PyG masks не використовуються.
- Batch формується вручну: конкатенація `x/edge_index/edge_attr`, окремий `batch_tensor` створюється як індекс графа для кожного вузла.

---

## 3. Process prediction (`core_GATConv_pr.py` як еталонний патерн)
### 3.1 Поля `Data`
- `x: FloatTensor [N_pad, F_node+1]` (`+1` = prefix active mask)
- `edge_index: LongTensor [2, E_pad]`
- `edge_attr: FloatTensor [E_pad, F_edge]`
- `y: LongTensor [1]` — індекс `next_node` у локальному `node_map`
- `doc_features: FloatTensor [F_doc]`
- `time_target: FloatTensor [1]` — `duration_work` next node
- `node_ids: list[str]` (включно з padding ids)
- `timestamps: FloatTensor [N_pad]`

### 3.2 Shape rules
- `N_pad = max_node_count` (із глобальної статистики).
- `E_pad = max_edge_count` (із глобальної статистики).
- Якщо ребер менше — доповнення нулями в `edge_index` і `edge_attr`.

### 3.3 Формула label `y`
`y = node_map[next_node]`, де:
- `current_nodes = executed[:i]`;
- `next_node = argmin_{node}(SEQUENCE_COUNTER_ > max(SEQUENCE_COUNTER_ у prefix))`.

### 3.4 Prefix representation
- `active_mask[j]=1.0`, якщо `node_j ∈ current_nodes`, інакше `0.0`.
- `timestamps[j]=START_TIME_(node_j)` для prefix вузлів, інакше `1.1`.
- `prefix_len` у метриках рахується як `sum(prefix_mask)` якщо є, інакше `sum(x[:, -1])`.

### 3.5 Маски
- Явна `prefix_mask` зазвичай не пишеться в `Data`; fallback іде через останню колонку `x`.

## 4. Batch structure (фактично)
Для train/val у `*_pr`:
- `x = cat(item.x)`
- `edge_index = cat(item.edge_index, dim=1)`
- `edge_attr = cat(item.edge_attr)`
- `doc_features = stack(item.doc_features)`
- `timestamps = cat(item.timestamps)`
- `batch_tensor = cat([full(num_nodes_i, i)])`
- `y_task = tensor([item.y.item()])`
- `y_time = stack(item.time_target).view(-1)`

Модель отримує псевдо-об'єкт `type("Batch", ..., {...})`, а не PyG `Batch`.
