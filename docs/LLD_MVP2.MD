# LLD_MVP2.MD

## 1. Topology Extractor
Алгоритм `TopologyExtractorService`:
1. Пройти по всіх трасах у `train_traces`.
2. Для кожної траси зібрати всі переходи $e_i \rightarrow e_{i+1}$.
3. Зберегти унікальні переходи у `allowed_edges`.
4. *Примітка:* Для імітації майбутніх версій ($\kappa+n$), TopologyExtractor може бути викликаний явно з наданням списку ребер (симуляція парсингу BPMN з Camunda).

## 2. Knowledge-Infusion Operator ($\Gamma$) у forward()
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
    # 1. Локальний контекст (MVP1)
    h_obs = self.baseline_gnn(contract.x, contract.edge_index)
    
    # 2. Глобальний контекст (MVP2)
    if contract.structural_edge_index is not None:
        h_struct = self.struct_gnn(contract.x, contract.structural_edge_index)
        # Злиття (Fusion)
        h = self.fusion_layer(torch.cat([h_obs, h_struct], dim=-1))
    else:
        # Fallback
        h = h_obs 
        
    return self.classifier(h)
```