# DATA_MODEL_MVP2.MD

## 1. Розширення існуючих DTO
Ми розширюємо об'єкти з `DATA_MODEL_MVP1.MD`, зберігаючи їхню сумісність.

### 1.1 RawTrace & PrefixSlice
- Додається обов'язкове поле `process_version: str` (за замовчуванням `"1"`, якщо відсутнє в логах).

### 1.2 GraphTensorContract (Оновлений)
Тензорний контракт отримує нові **Optional** поля для нормативного графа:
```python
class GraphTensorContract(BaseModel):
    # MVP1 (Залишаються обов'язковими)
    x_cat: Tensor
    x_num: Tensor
    edge_index: Tensor
    edge_type: Tensor
    batch: Tensor
    y: Tensor
    
    # MVP2 (Нові поля для EOPKG)
    structural_edge_index: Optional[Tensor] = None
    structural_edge_weight: Optional[Tensor] = None # Для статистики (частота/час), якщо доступно
    version_emb_idx: Optional[Tensor] = None # Індекс версії \kappa
```

## 2. Нові DTO для EOPKG
### 2.1 ProcessStructureDTO
```python
class ProcessStructureDTO(BaseModel):
    version: str
    allowed_edges: List[Tuple[str, str]] # Список дозволених переходів (activity_id -> activity_id)
    edge_statistics: Dict[Tuple[str, str], Dict[str, float]] # Опціональна статистика
```