# LLD_MVP2.MD

## 1. Topology Extractor
**Алгоритм `TopologyExtractorService`:**
1. Пройти по всіх трасах у `train_traces`.
2. Для кожної траси зібрати всі переходи $e_i \rightarrow e_{i+1}$.
3. Зберегти унікальні переходи у `allowed_edges`.
4. *Примітка:* Для імітації майбутніх версій ($\kappa+n$), TopologyExtractor може бути викликаний явно з наданням списку ребер (симуляція парсингу BPMN з Camunda).

## 2. DynamicGraphBuilder: Mask Generation Algorithm
Для забезпечення `Tensor Purity` і швидкого розрахунку OOS, `DynamicGraphBuilder` повинен конвертувати нормативні правила у тензор маски для кожного графа.

**Алгоритм для одного графа (префіксу):**
1. Створити `allowed_mask = torch.zeros(C, dtype=torch.bool)` (де $C$ — кількість унікальних класів/активностей).
2. Визначити останню подію префіксу: `last_event = prefix.prefix_events[-1]`.
3. Отримати список дозволених наступних подій з `ProcessStructureDTO`: 
   `allowed_targets = [dst for src, dst in dto.allowed_edges if src == last_event.activity_id]`
4. Для кожного `target` у `allowed_targets`:
   - Отримати його числовий індекс через словник енкодера (наприклад, `class_idx = encoder.get_class_index(target)`).
   - Встановити `allowed_mask[class_idx] = True`.
5. Покласти цей тензор `allowed_mask` (shape: `[C]`) у `GraphTensorContract` як `allowed_target_mask`.
*(Під час формування батчу PyG DataLoader автоматично об'єднає їх у тензор `[B, C]`)*.

## 3. Knowledge-Infusion Operator ($\Gamma$) у forward()
Модель `EOPKGFusionGNN` має обробляти ситуацію, коли `structural_edge_index` є `None`.

### Математична модель (Fallback Logic)
Нехай $H^{(l)}$ — приховані стани вузлів на шарі $l$.
Якщо `structural_edge_index` $\neq \text{None}$:
$$H^{(l+1)} = \text{GNN}_{obs}(H^{(l)}, \text{edge\_index}_{obs}) + \alpha \cdot \text{GNN}_{struct}(H^{(l)}, \text{edge\_index}_{struct})$$
(або використання `structural_edge_index` як маски для Attention).

Якщо `structural_edge_index` $== \text{None}$ (Fallback до MVP1):
$$H^{(l+1)} = \text{GNN}_{obs}(H^{(l)}, \text{edge\_index}_{obs})$$

### Псевдокод forward:
```python
def forward(self, contract: GraphTensorContract) -> Tensor:
    # 1. Локальний контекст
    h_obs = self.baseline_gnn(contract.x, contract.edge_index)
    
    # NEW: 2. Латентне кодування версії (H4)
    v_emb = None
    if contract.version_emb_idx is not None:
        v_emb = self.version_embedding(contract.version_emb_idx)
    
    # 3. Глобальний контекст та злиття (H1)
    if contract.structural_edge_index is not None:
        h_struct = self.struct_gnn(contract.x, contract.structural_edge_index)
        
        # Злиття локального, структурного та версійного контекстів
        fusion_input = [h_obs, h_struct]
        if v_emb is not None: fusion_input.append(v_emb)
        
        h = self.fusion_layer(torch.cat(fusion_input, dim=-1))
    else:
        h = h_obs # Fallback
        
    return self.classifier(h)
```

## 4. Evaluator Metrics & Slicing Algorithm (H1, H4, H5)

### 4.1 Підрахунок OOS (Out of Structure)
В `Evaluator` метрика OOS рахується виключно тензорними операціями без звернення до словників:
Нехай $B$ — розмір батчу, `y_hat` — індекси передбачених класів (shape `[B]`), а `allowed_mask` — маска з контракту (shape `[B, C]`).
```python
if contract.allowed_target_mask is not None:
    # Витягуємо значення маски (True/False) для передбачених класів
    is_allowed = contract.allowed_target_mask[torch.arange(B), y_hat]
    # OOS - це частка False (недозволених) прогнозів
    oos_rate = (~is_allowed).float().mean().item()
```

### 4.2 Алгоритм зрізів (Slicing)
Для розрахунку зрізів (slices), `Evaluator` повинен на етапі агрегації батчу:
1. Витягти `process_version` з `PrefixSlice` (через мапінг або передачу атрибутів у батчі).
2. Визначити довжину графа $L = \text{contract.num_nodes}$ (кількість подій у префіксі).
3. Розбити довжини на біни (наприклад, `L <= 5`, `5 < L <= 10`, `L > 10`).
4. Формувати окремі ключі у результуючому словнику метрик:
   - `test_f1_v1`, `test_f1_v2`...
   - `test_f1_len_1_5`, `test_f1_len_6_10`...