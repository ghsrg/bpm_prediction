# LLD.MD (Low-Level Design)

## 1. Статус
`APPROVED` (Scope: MVP1 - Baseline GNN)

## 2. Призначення
Специфікація математичного ядра системи для фази MVP1. Описує стратегії нарізки префіксів, вектеризації ознак (Feature Engineering), формування тензорів та базової архітектури нейромережі.

**Науковий контекст (Baseline Constraint):** Цей дизайн імплементує виключно мікро-рівневий енкодер ($\Phi_{micro}$) на емпіричних логах. Глобальний макро-енкодер ($\Phi_{macro}$), оператор генеративного злиття ($\Gamma$), структурний регуляризатор ($\mathcal{L}_{reg}$) та семафор надійності ($S$) навмисно **відключені** у цій версії для формування чистого Baseline-результату, з яким буде порівнюватись цільова методологія у наступних експериментах.

---

## 3. Prefix Generation (`PrefixPolicy`)

Алгоритм підготовки навчальної вибірки базується на стратегії **All Prefixes** (генерація префіксів для кожного кроку процесу).

### 3.1. Математика нарізки
Для кожної траси $\sigma = \langle e_1, e_2, ..., e_n \rangle$ генерується множина префіксів:
- Цикл $k$ від $1$ до $n-1$.
- Вхідний префікс (історія спостереження): $\sigma_{[1:k]} = \langle e_1, ..., e_k \rangle$.
- Цільова подія (Target): $y = e_{k+1}$.

### 3.2. Обробка крайових випадків
- Траси довжиною $n < 2$ відкидаються.
- Префікси не доповнюються нулями (No Padding). PyTorch Geometric обробляє графи різного розміру через блоково-діагональні матриці (`Batch`).

---

## 4. Feature Engineering ($\Phi_{enc}$ для мікро-рівня)

Перетворення `EventRecord` у числові тензори $X_V$ для графа екземпляра $G_{obs}$.

### 4.1. Категоріальні ознаки (Categorical)
- `activity_id` $\to$ індекс зі словника $\to$ `nn.Embedding(num_activities, dim_act)`.
- `resource_id` $\to$ індекс зі словника $\to$ `nn.Embedding(num_resources, dim_res)`.
- Невідомі значення (OOV/OOD) мапляться в індекс `0` (зарезервовано для `<UNK>`).

### 4.2. Неперервні (Continuous) та Часові ознаки
Часові дельти: `duration`, `time_since_case_start`, `time_since_previous_event`.
- **MVP1 (Linear Normalization):** Обов'язкова Z-score стандартизація $x_{norm} = \frac{x - \mu}{\sigma}$ на основі глобальної статистики тренувального датасету (етап `StabilizeAndNormalize`).
- *(Future Scope: Гармонійне кодування Time2Vec $\Phi_{time}$ закладено в цільовій архітектурі, але у MVP1 використовуються Z-нормалізовані скаляри).*

### 4.3. Злиття ознак (Node Feature Vector)
Для кожного вузла $v \in V_\sigma$ формується локальний вектор:
$X_V[v] = [E_{act}(act\_id) \parallel E_{res}(res\_id) \parallel t_{duration} \parallel t_{case} \parallel t_{prev}]$
Розмірність вузла: `d_V = dim_act + dim_res + 3`.

---

## 5. Graph Tensor Construction

Перетворення префікса у формат `GraphTensorContract` (сумісний з PyG `Data`).

### 5.1. Топологія графа екземпляра ($G_{obs}$)
Для MVP1 емпіричний граф будується як лінійна послідовність (DFG-ланцюг без EPOKG-зв'язків):
- Вузли ($V_\sigma$): хронологічний порядок подій у префіксі.
- Ребра ($A$ / `edge_index`): орієнтований зв'язок від $e_i$ до $e_{i+1}$.
- `edge_type`: у MVP1 всі ребра мають єдиний базовий тип (observed_flow).

### 5.2. Структура `GraphTensorContract`
Уніфікований DTO для передачі графа в нейромережу.
```python
from typing import TypedDict
import torch

class GraphTensorContract(TypedDict):
    x: torch.FloatTensor          # X_V: [N_nodes, d_V]
    edge_index: torch.LongTensor  # A: [2, E_edges]
    edge_type: torch.LongTensor   # [E_edges] (Для MVP1 заповнено нулями)
    y: torch.LongTensor           # [1] (activity_id цільової події e_{k+1})
    batch: torch.LongTensor       # B: [N_nodes] (індекси батчування від PyG)
```

---

## 6. Baseline GNN Architecture (Мікро-енкодер $\Phi_{micro}$)

Базова архітектура прямого поширення ($f_\theta$) для тестування гіпотези. 
*Відсутні $z_{plan}^{(\kappa)}$, $\Gamma$-оператор та макро-компоненти.*

### 6.1. Pipeline шарів
1. **Input Layer:** Трансформація сирих векторів $X_V$ у прихований простір $\mathbb{R}^{d}$.
2. **Message Passing ($\phi$):**
   - 2-3 шари Graph Convolution (наприклад, GraphSAGE або GATv2Conv).
   - Активація: ReLU або GELU.
   - Dropout: $0.2 - 0.3$.
   - Обчислення прихованих станів: $H^{(l+1)} = \phi(H^{(l)}, A)$.
3. **Readout Layer ($\rho$):**
   - Агрегація графа в єдиний вектор екземпляра $h_{obs}$: 
   - $h_{obs} = \rho(H^{(L)}) = \text{global\_mean\_pool}(H^{(L)}, B)$.
4. **Task Head ($g$):**
   - Проєкційний MLP-шар: $\hat{y} = \text{Softmax}(W_{out} h_{obs} + b_{out})$.
   - Розмірність виходу: `[num_activities]`.

### 6.2. Контракт інтерфейсу
Усі моделі імплементують єдиний порт:
```python
class BaselineGNN(IGNNModelPort):
    def forward(self, data: GraphTensorContract) -> torch.Tensor:
        # Повертає логіти розміром [batch_size, num_classes]
        ...
```

### 6.3. Цільова функція (Optimization)
- **Функціонал втрат:** $\mathcal{L}_{total} = \mathcal{L}_{task}$ (Standard Cross-Entropy Loss). *Структурний регуляризатор $\beta \mathcal{L}_{reg}$ у MVP1 дорівнює нулю.*
- **Метрики оптимізації:** Macro F1 (основна для дисбалансованих логів), Accuracy.
- **Оптимізатор:** Adam / AdamW з класичним градієнтним спуском (Mini-batch SGD).