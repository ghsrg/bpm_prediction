# 0. Мета документа

Цей документ визначає архітектурне бачення, технічні вимоги та рамки проєкту **bpm_prediction**.  
Документ слугує базою для розробки, масштабування, інтеграцій та проведення наукових експериментів.

---

## 0.1. Призначення системи

`bpm_prediction` розробляється як універсальна платформа для аналізу бізнес-процесів на основі:

- графових нейронних мереж (GNN) - основна задача,

в режимі сумісності для порівняння:
- трансформерів,
- часових моделей,
- інших архітектур.

Система призначена для:

- прогнозування наступних активностей чи всього наступного графа,
- прогнозування часу виконання задачі або всього маршруту,
- виявлення аномалій,
- порівняння Logs-only / BPMN-aware / POKG-enhanced підходів.

---

## 0.2. Практична цінність

Система виконує роль промислового **Proof-of-Concept для Camunda/BPMS**, та наукового порівняння та повторюваності на відкритих наборах даних, забезпечуючи можливість:

- працювати з логами бізнес-процесів у реальному середовищі,
- формувати Instance Graph (IG),
- інтегрувати BPMN-моделі,
- створювати та використовувати Process Organizational Knowledge Graph (POKG),
- виконувати inference у напівреальному часі.

---

## 0.3. Наукова цінність

Платформа підтримує повний цикл підготовки даних і експериментів:

- вивчення впливу структури BPMN/POKG з версійністю та даних організації, політик на точність моделей,
- порівняння GNN з MLP, Transformer та інших архітектур,
- дослідження ефекту довжини префікса (prefix length),
- дослідження впливу технічних гейтів та вузлів BPMN
- аналіз out-of-scope класифікацій,
- робота з відкритими наборами (BPI Challenge, XES),
- проведення статистично значущих тестів (paired t-test) та представлення наукових результатів у вигляді діаграм та табилць.

---

## 0.4. Візуалізація експериментів

Система автоматично генерує широкий набір аналітичних артефактів в режимі навчання та тестування

- графіки навчання, метрик і префіксів,
- confusion matrices, aggregated matrices,
- regression curves, trend lines,
- деталізовані діаграми (mean ± std, min/max) 
- heatmaps,
- дашборди точності/часу для архітектур,
- візуалізації IG/BPMN/POKG,
- діаграми архітектур моделей,
- розподіли даних, вузлів, ребер, префіксів, та їх атрибутів.

Утиліти для цього зібрано в `src/utils/` (див. розділ 9).

---

## 0.5. Універсальність та модульність

Платформа дозволяє підключати будь-які джерела даних через спеціалізовані адаптери:

- Camunda через MSSQL,
- XES-логи з файлів,
- REST API для промислового PoC,


Автоматично формується:

- Instance Graph (IG),
- BPMN-граф,
- Fusion Graph (IG + BPMN + POKG),
- тренувальні набори,
- фічі для моделей,
- статистичні звіти,
- діаграми та аналіз.

---

## 0.6. Архітектурні цілі

Система повинна відповідати принципам:

- **Clean Architecture**,  
- **Separation of Concerns**,  
- **Low Coupling & High Cohesion**,  
- **Reproducibility**,  
- **Explicit Configuration**,  
- **Modular Design**,  
- **Scalability**.

Основні вимоги:

- строгий поділ на шари (adapters → graph → models → pipelines → utils),
- уніфікований інтерфейс моделей, приймає динамічний набор даних та фіч вказаних в конфігах,
- окремі конфігурації для кожного бізнес-процесу,
- повний seed-контроль,
- підтримка best/last моделей,
- робота в експериментальному та продуктивному режимах.

---

## 0.7. Для кого цей документ

Документ призначений для:

- розробників,
- архітекторів,
- дослідників ML/BPM,
- інтеграторів BPMS,
- авторів наукових публікацій.

---

## 0.8. Роль документа

Документ слугує:

1. архітектурним описом (README.md),
2. основою для Technical Design Document,
3. орієнтиром для Codex-промптів та генерування коду,
4. фундаментом для рефакторингу та побудови нової архітектури,
5. базою для наукових експериментів.


# 1. High-Level Design (HLD): Архітектура системи

Архітектура `bpm_prediction` побудована відповідно до принципів **Clean Architecture** та **Modular ML Systems**, з чітким розмежуванням відповідальностей між шарами.  
Цей розділ визначає глобальну структуру системи на рівні концепцій, компонентів і потоків даних (HLD).  
Деталізація класів, методів і структур даних буде подана у Low-Level Design (LLD).

---

## 1.1. Архітектурні шари

Система складається з п’яти основних логічних шарів:

### **1. Data Adapters Layer**
Відповідає за уніфікований доступ до джерел даних:
- MSSQL (Camunda, BPMS)
- XES-файли (BPI Challenge тощо)
- REST API
- локальні структури

Адаптери повертають **dataframes у стандартизованому форматі**, не містять логіки графів або ML.

---

### **2. Graph Construction Layer**
Формує єдине графове представлення процесів:

- **Instance Graph (IG)**  
  Побудований з логів виконання процесів.

- **BPMN Graph**  
  Побудований зі структур BPMN-моделі (gateway, flow, events, tasks).

- **Fusion Graph**  
  Комбінація IG + BPMN + POKG (Process Organizational Knowledge Graph).
  Використовується для структурування даних у режимах BPMN-aware та POKG-enhanced.

Графи завжди приводяться до єдиного формату (networkx або torch_geometric).

---

### **3. Model Layer**
Містить реалізації моделей:

- GNN: GCN, GAT, GraphSAGE, GGNN, MixHop, MuseGNN
- Temporal GNN: TGAT, TemporalGAT, T-GCN
- Transformer-based моделі
- Multi-task Predictors (activity + time)
- Модулі для anomaly detection (AE, GNN-AE)
- Підтримка різних heads:
  - класифікація
  - регресія
  - multi-output

Всі моделі успадковують базовий інтерфейс `BaseModel`:
prepare_data(...)
build(...)
train(...)
evaluate(...)
predict(...)


---

### **4. Pipelines Layer (Orchestration)**
Відповідає за послідовну та модульну обробку:

- preprocess_pipeline  
- doc_analysis_pipeline  
- generate_variations_pipeline  
- train_pipeline  
- train_predict_pipeline  
- evaluate_pipeline  
- clustering_pipeline  

Пайплайни не містять ML-логіки — лише керування та orchestration.

---

### **5. Utilities Layer**
Підтримуючий функціональний шар:

- file_utils
- graph_utils
- diagrams
- visualizer
- metrics
- logger
- string_utils
- graph_creator
- graph_variations

Цей шар не повинен містити доменної логіки або взаємодій між модулями.

---

## 1.2. Глобальний потік даних у системі

### **(1) Завантаження даних**
Джерело → Adapter → Standard DataFrame

MSSQLAdapter
XESAdapter
RESTAdapter

---

### **(2) Формування графів**
Дані подій → IG Builder  
BPMN XML → BPMN Builder  
POKG / оргструктура → Fusion Builder

Результат: **графи у форматі PyTorch Geometric / networkx**

---

### **(3) Формування ознак (Features)**
Feature Builder готує:

- node features
- edge features
- global features
- time2vec / positional encodings
- embeddings документів
- one-hot / categorical encodings

Використовує конфігурацію у `features.yaml`.

---

### **(4) Навчання моделей**
Пайплайни забезпечують:

- batch generator
- early stopping
- збереження best/last моделей
- контроль seed (повторюваність експериментів)
- логування метрик
- генерацію діаграм
- аналіз prefix-length

---

### **(5) Оцінка (Evaluation)**
Генерується:

- confusion matrix
- top-1 / top-3 accuracy
- regression curves
- prefix bin analysis
- BPMN-aware vs logs-only різниці
- t-tests та статистичні порівняння
- візуалізації IG/BPMN/POKG

---

### **(6) Інференс**
Формується IG → Features → Завантаження best-моделі → Прогноз:

- наступної активності
- часу виконання
- аномалії

---

## 1.3. Архітектурні інваріанти (обов’язкові правила)

1. Моделі **не мають звертатись до файлової системи** напряму.  
2. Всі конфігурації винесені у YAML і не хардкодяться в коді.  
3. Графи IG/BPMN/Fusion мають **єдину структуру** та поведінку.  
4. Пайплайни не знають конкретної архітектури моделі.  
5. Повна підтримка “model-per-process”:
   - свої фічі  
   - свої конфіги  
   - свої моделі  
   - свій seed  
6. Режими logs/bpmn/pokg повинні працювати в одному пайплайні.  
7. Всі візуалізації зберігаються структуровано за схемою:
output/
<process>/
<model>/
<mode>/
<seed>/
diagrams/
confusion_matrices/
prefix_analysis/
checkpoints/


---

## 1.4. Масштабування і майбутнє розширення

Архітектура дозволяє безболісно додавати:

- нові адаптери даних (ERP, CRM, Jira)
- нові графові представлення (DFG, SRL, OrgGraph)
- нові моделі GNN/Transformer
- нові сценарії експериментів
- inference-сервіс (FastAPI)
- витягування знань у POKG

---






# 2. High-Level Design (HLD): Data Adapters Layer

## 2.1. Цілі Data Adapters Layer
Data Adapters Layer відповідає за уніфікований доступ до джерел сирих даних. Мета — отримати стандартизовані DataFrame та метадані, не змішуючи логіку графів або ML.

## 2.2. Поточні компоненти
- mssql_connector.py — SQL-запити до MSSQL.
- rest_api_connector.py — REST-заглушка.
- data_processing.py — базовий препроцесинг (у майбутньому винести).

## 2.3. Цільовий інтерфейс адаптера
class BaseDataAdapter:
    load_events(...)
    load_bpmn(...)
    load_metadata(...)

Конкретні реалізації:
- MSSQLAdapter
- XESAdapter
- RESTAdapter
- MockAdapter

## 2.4. Вихідний формат
### Event Log DataFrame
Обов’язкові колонки: case_id, activity, timestamp, resource, role, bpmn_element_id, lifecycle, attributes.

### BPMN Data
Структура BPMN: nodes, flows, attributes, mapping.

### Metadata
Довідники та додаткові таблиці.

## 2.5. Зв’язок з Utilities Layer
Дозволяється використовувати:
- logger
- file_utils_l
- save/load parquet/hdf5

Забороняється:
- graph_utils
- graph_creator
- core-моделі

## 2.6. Конфігурація
Через config/data_sources.yaml:
- connection strings
- endpoint-и
- фільтри
- обмеження вибірки

## 2.7. Seed
Використовується лише для sampling.

## 2.8. Розширення
### MSSQL / Camunda
MSSQLAdapter з уніфікованим форматом.

### XES
XESAdapter: конвертація у стандартний Event Log DataFrame.

### REST
RESTAdapter: робота з JSON API.

## 2.9. Результат
- Єдиний інтерфейс доступу до даних.
- Повна ізоляція джерел від графів та моделей.
- Легке додавання нових адаптерів.



# 3. High-Level Design (HLD): Graph Construction Layer

## 3.1. Призначення Graph Construction Layer
Graph Construction Layer відповідає за перетворення сирих даних подій, BPMN-моделей та корпоративних знань у стандартизовані графові структури. Це центральний шар, який формує основу для GNN/Transformer моделей.

Цілі:
- побудова Instance Graph (IG) із логів процесів;
- побудова BPMN Graph зі структурних моделей;
- побудова Fusion Graph, який поєднує IG + BPMN + POKG;
- забезпечення уніфікованого формату графів, сумісного з PyTorch Geometric.

---

## 3.2. Основні компоненти шару

### 3.2.1. InstanceGraphBuilder
Відповідає за побудову графу виконання процесу (IG) з event log:

- вузли = події або агреговані активності;
- ребра = фактичні переходи за SEQUENCE_COUNTER_;
- атрибути вузлів:
  - activity name,
  - timestamp (absolute / relative),
  - duration,
  - resource/role,
  - статус,
  - document metadata,
  - numeric/categorical features.

Функціонал:
- сортування подій за SEQUENCE_COUNTER_;
- створення ребер (u → v);
- time2vec / temporal embeddings;
- глобальні ознаки (prefix length, graph depth, path stats).

---

### 3.2.2. BPMNGraphBuilder
Будує структурний граф із BPMN XML:

- вузли = tasks, gateways, events;
- ребра = sequenceFlows;
- атрибути:
  - type,
  - label,
  - lane/role,
  - is_subprocess,
  - structural position.

Функціонал:
- парсинг BPMN XML,
- вилучення типів елементів,
- створення структурних ребер,
- мапінг BPMN element → event log activity.

---

### 3.2.3. FusionGraphBuilder
Об'єднує IG + BPMN Graph + Knowledge Graph (POKG):

Fusion Graph = IG ⊕ BPMN ⊕ POKG

Типи інтеграції:
- node-level fusion,
- edge-level fusion,
- semantic fusion (ембеддинги з POKG),
- multi-relational graph.

---

## 3.3. Інтерфейс GraphBuilder

```python
class BaseGraphBuilder(ABC):

    @abstractmethod
    def build(self, data, **kwargs):
        pass

    @abstractmethod
    def to_pyg(self, graph):
        pass
```

---

## 3.4. Формати графів

### NetworkX Graph
- внутрішній формат для обробки та перевірок.

### PyTorch Geometric Data
- формат для моделі:
  - x (node features),
  - edge_index,
  - edge_attr,
  - global_features,
  - y (labels).

---

## 3.5. Генерація фіч

Feature Builder забезпечує:
- нормалізацію числових ознак,
- one-hot кодування,
- time2vec,
- BPMN структурні фічі,
- документні ембеддинги.

Конфіги: features.yaml, preprocessing.yaml

---

## 3.6. Prefix-Length режими

GraphBuilder підтримує:
- prefix-cut за кількістю подій,
- time-cut,
- dynamic cut (для незавершених інстансів).

Результат — окремі IG/Fusion Graph для кожного префікса.

---

## 3.7. Виклик з Pipelines Layer

```
events = adapter.load_events(process_id)
bpmn = adapter.load_bpmn(process_id)

ig = InstanceGraphBuilder.build(events)
bg = BPMNGraphBuilder.build(bpmn)
fusion = FusionGraphBuilder.build(ig, bg, pokg)
pyg_graph = FusionGraphBuilder.to_pyg(fusion)
```

---

## 3.8. Обмеження

- Graph Layer не викликає ML-моделі,
- не працює з файловою системою,
- вся конфігурація — через YAML,
- побудова графів детермінована при одному seed.

---

## 3.9. Результат

Після реалізації система отримає:
- уніфіковану побудову IG/BPMN/Fusion Graph,
- можливість POKG-aware навчання,
- відокремленість доменної логіки,
- повторювані графи для експериментів,
- розширюваність для нових типів графів.


# 4. High-Level Design (HLD): Model Layer

## 4.1. Призначення Model Layer
Model Layer — це незалежний шар, який містить реалізації всіх моделей, що використовуються для:
- прогнозування наступної активності,
- прогнозування часу виконання,
- anomaly detection,
- multi-task прогнозування (activity + time),
- порівняння Logs/BPMN/POKG режимів.

Моделі повністю відокремлені від файлової системи, пайплайнів та джерел даних.

---

## 4.2. Архітектурні принципи Model Layer

1. **Єдиний базовий інтерфейс** `BaseModel`  
2. **Повна ізоляція ML від I/O**  
3. **Підтримка кількох типів висновків (класифікація, регресія, multi-output)**  
4. **Сумісність з Prefix-Length режимами**  
5. **Гнучка адаптація під Fusion Graph / BPMN Graph / IG**  
6. **Можливість додавати нові архітектури без зміни пайплайнів**  
7. **Контроль seed для повторюваності**

---

## 4.3. Інтерфейс BaseModel

```python
class BaseModel(ABC):

    @abstractmethod
    def prepare_data(self, graphs, **kwargs):
        pass

    @abstractmethod
    def build(self, input_dim, **kwargs):
        pass

    @abstractmethod
    def train(self, train_data, val_data, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, test_data, **kwargs):
        pass

    @abstractmethod
    def predict(self, graph, **kwargs):
        pass
```

---

## 4.4. Підтримувані моделі

### 4.4.1. Graph Neural Networks (GNN)

- GCN  
- GAT / GATv2  
- GraphSAGE  
- APPNP  
- GGNN  
- DeepGCN  
- MixHop  
- MP-GCN  
- EvolveGCN (future extension)  
- MuseGNN  
- Graphormer (graph transformer)  

Всі GNN працюють із `torch_geometric.data.Data`.

---

### 4.4.2. Temporal GNN

- TGAT  
- Temporal GAT  
- T-GCN  
- GRU-GNN / LSTM-GNN  
- Time-aware Graph MixHop  

Використовують:
- time2vec,
- sinusoidal encoding,
- absolute/relative time.

---

### 4.4.3. Transformer-based Models

- Encoder-only трансформери,
- Multi-head attention графових ознак,
- Hybrid GNN+Transformer моделі.

---

### 4.4.4. Класичні моделі

Для anomaly detection / baseline:
- MLP  
- CNN  
- RNN  
- Autoencoder  
- Graph Autoencoder  
- Variational Autoencoder (опційно)

---

## 4.5. Heads (вихідні шари)

### 4.5.1. Activity Classification Head
- softmax
- top-1 / top-3 accuracy
- OOS-дослідження

### 4.5.2. Time Regression Head
- MAE / RMSE
- логарифмована регресія (опція)

### 4.5.3. Multi-task Head
Один backbone → два виходи:
- activity logits,
- time prediction.

---

## 4.6. Підготовка даних (prepare_data)

Функції моделей не мають працювати з файлами — тільки з графами:

- node features → `x`
- edge features → `edge_index`, `edge_attr`
- global features  
- document embeddings  
- bpmn structural encodings  
- target (`y_class`, `y_time`)

Форматується відповідно до конфігів у:
- `features.yaml`
- `model.yaml`

---

## 4.7. Навчання (train)

Навчання включає:

- batch loader
- early stopping
- оптимізатор (AdamW / Adam / RAdam)
- scheduler (опційно)
- логування loss/metrics
- збереження best/last checkpoints

---

## 4.8. Оцінка (evaluate)

Тестування повинно повертати:

- Accuracy, Top-3
- F1 / Precision / Recall
- AUPRC, ROC-AUC
- ADR / FAR / FNR
- Regression MAE/RMSE
- prefix-length evaluation
- confusion matrices
- distributions

---

## 4.9. Інференс (predict)

Процес:
1. Отримати граф → pyg_graph  
2. Прогнати forward  
3. Повернути:
   - top-k activities,
   - time prediction,
   - anomaly score.

---

## 4.10. Конфігурація моделей

Через `model.yaml`:
- тип моделі (GCN/GAT/TGAT/Transformer/…)
- параметри:
  - hidden_dim
  - num_layers
  - dropout
  - heads
  - attention type
  - time2vec dim
  - pooling type
  - optimizer
  - learning_rate
  - loss weights (α для time/head)
- preprocessing flags (normalize, standardize)

---

## 4.11. Залежність від інших шарів

Model Layer залежить лише від:
- PyTorch / PyTorch-Geometric
- Feature Builder (обмежено)
- конфігів (YAML)
- logger

Model Layer **не взаємодіє**:
- з файлами,
- із pipelines,
- з adapters,
- з graph builders.

---

## 4.12. Результат впровадження

Після реалізації:
- всі моделі мають єдиний API,
- додавання нової архітектури не потребує змін у пайплайнах,
- спрощується експериментування,
- модель легко переносити у production inference,
- можлива автоматична генерація діаграм архітектури.




# 5. High-Level Design (HLD): Pipelines Layer

## 5.1. Призначення Pipelines Layer

Pipelines Layer — це шар оркестрації. Він визначає *сценарії роботи* системи, використовуючи:

- Data Adapters (джерела даних)
- Graph Construction Layer (побудова IG/BPMN/Fusion)
- Model Layer (навчання / тестування / інференс)
- Utilities Layer (логування, візуалізація, статистики)

Особливість шару: він **не містить ML-логіки**.  
Пайплайни лише координують незалежні модулі.

---

## 5.2. Основні сценарії (модулі)

### 5.2.1. Preprocess Pipeline
Призначення:
- завантаження сирих даних через адаптери;
- фільтрація, cleaning;
- кешування у parquet/hdf5;
- підготовка даних для побудови графів.

### 5.2.2. Doc Analysis Pipeline
- збагачує логи документною інформацією;
- виконує групування за ROOT_PROC_INST_ID / case_id;
- формує структури для Instance Graph Builder.

### 5.2.3. Generate Variations Pipeline
- створення нормальних та аномальних варіацій графів;
- augmentations:
  - зміна timestamp,
  - noise,
  - document mutations,
  - edge rewiring;
- використовується для anomaly detection.

### 5.2.4. Train Pipeline
Призначення:
- навчання моделей anomaly detection / GNN;
- split train/val/test;
- batch processing;
- early stopping;
- збереження best/last checkpoint;
- побудова діаграм learning curves.

### 5.2.5. Train Predict Pipeline
Цей пайплайн тренує моделі для задач:
- прогноз наступної активності,
- прогноз часу,
- multi-task архітектур.

Особливості:
- мультиархітектурна підтримка через MODEL_MAP;
- робота з prefix-length експериментами;
- автоматична генерація:
  - confusion matrices,
  - prefix-bins statistics,
  - regression curves.

### 5.2.6. Evaluate Pipeline
Виконує:
- тестування моделей;
- збір статистики по архітектурах;
- t-tests Logs vs BPMN;
- підготовку даних для публікацій.

### 5.2.7. Clustering Pipeline
- кластеризація графів/трас;
- silhouette-аналіз;
- візуалізація embedding space.

---

## 5.3. Принцип роботи пайплайнів

Типовий цикл:

```
events = adapter.load_events()
bpmn = adapter.load_bpmn()

graphs = GraphBuilder.build(...)
features = FeatureBuilder.prepare(...)

model = ModelFactory.create(...)
model.train(...)

evaluation = model.evaluate(...)
visualizer.plot(...)
```

---

## 5.4. Вимоги до Pipelines Layer

### 5.4.1. Нульова залежність у зворотний бік
Моделі не знають про пайплайни.

### 5.4.2. Мінімум конфігурації в коді
Всі параметри — через YAML:
- режим logs/bpmn/pokg;
- архітектура моделі;
- набір фіч;
- learning rate, batch size, hidden_dim;
- параметри графів (prefix, time windows).

### 5.4.3. Повторюваність експериментів
- seed,
- контроль датасетів,
- збереження усіх конфігів при старті експерименту.

### 5.4.4. Уніфікований рендер результатів
Пайплайни зобов’язані викликати:
- confusion matrix visualizer,
- графіки навчання,
- prefix statistics,
- architecture radar charts.

---

## 5.5. Взаємодія з файловою системою

Пайплайни відповідають за:
- створення output-директорій,
- збереження чекпоінтів,
- збереження діаграм,
- логування,
- експорт статистик.

Структура:

```
output/
  process_name/
    model_name/
      mode/
        seed/
          checkpoints/
          diagrams/
          metrics/
          prefix/
          confusion/
```

---

## 5.6. Обмеження (інваріанти)

- Пайплайни не містять графової логіки.
- Пайплайни не містять ML-алгоритмів.
- Пайплайни не працюють з DataFrame — лише з Adapter API.
- Пайплайни не змінюють результатів моделей — лише викликають їх.
- Пайплайни мають бути idempotent (повторний запуск без помилок).

---

## 5.7. Результат впровадження

Після реалізації:
- система матиме модульні та масштабовані сценарії;
- експерименти стануть керованими, повторюваними та структурованими;
- додавання нової архітектури моделі не потребуватиме змін у пайплайнах;
- зросте якість експериментальних результатів для наукових статей.



# 6. High-Level Design (HLD): Utilities Layer

## 6.1. Призначення Utilities Layer

Utilities Layer — це службовий шар, що містить набір допоміжних функцій, які використовують:
- пайплайни,
- графові модулі,
- модельний шар,
- аналітичні компоненти,
- візуалізацію.

Цей шар **не містить доменної логіки** та не має знати:
- нічого про ML-архітектури,
- деталей графів,
- структури адаптерів,
- специфіки пайплайнів.

Його мета — створити стабільний, незалежний набір функцій, які можна використовувати у будь-якому модулі.

---

## 6.2. Підсистеми Utilities Layer

Utilities Layer складається з таких компонентів:

### 6.2.1. File Utilities
Модулі:
- `file_utils.py`
- `file_utils_l.py`

Функціонал:
- читання/запис parquet/hdf5/csv,
- управління шляхами,
- збереження/читання графів (networkx, pyg),
- зберігання чекпоінтів,
- підготовка директорій,
- об'єднання статистичних файлів,
- генерація індексів файлів,
- функції для Excel/JSON.

Головне правило:
- **будь-який доступ до файлової системи повинен іти через цей модуль.**

---

### 6.2.2. Logger
- кольоровий логер,
- форматовані події,
- відображення DataFrame/графів,
- підтримка debug/info/warning/error,
- можливість логування у файл,
- конфігурація через YAML.

Logger використовується у всіх шарах.

---

### 6.2.3. Metrics
Модуль `metrics.py` забезпечує обчислення:
- AUPRC,
- ROC-AUC,
- F1/Precision/Recall,
- ADR,
- FAR/FPR/FNR,
- метрик регресії.

Особливість: модельний шар не обчислює метрики сам — тільки цей модуль.

---

### 6.2.4. Graph Utilities
Модуль `graph_utils.py`:
- чистка графів,
- інспекція структури,
- нормалізація атрибутів,
- форматування числових/категоріальних ознак,
- перетворення документних атрибутів.

Модуль не знає про ML або пайплайни — лише про структуру графа.

---

### 6.2.5. Graph Creation Helpers
Модуль `graph_creator.py`:
- парсинг BPMN,
- побудова графів процесів,
- пошук ключових елементів BPMN.

Використовується у Graph Construction Layer.

---

### 6.2.6. Graph Variations
Модуль `graph_variations.py`:
- аугментації графів,
- створення варіацій,
- генерування аномалій,
- перевірка коректності графів.

Цей модуль потрібен для anomaly detection експериментів.

---

### 6.2.7. Visualizer & Diagrams
`visualizer.py` та `diagrams.py` забезпечують:

- learning curves,
- regression plots,
- confusion matrices,
- prefix-length charts,
- heatmaps,
- radar charts,
- порівняння архітектур,
- графічне представлення моделей (GNN/CNN/RNN/Transformer diagrams),
- візуалізацію графів (plotly, dot, heatmap).

Важливо: візуалізатор працює тільки з результатами, а не з моделями напряму.

---

### 6.2.8. String Utilities
`string_utils.py`:
- форматування дат,
- перетворення у JSON-серіалізовані типи,
- допоміжні конвертори.

---

## 6.3. Обмеження та інваріанти Utilities Layer

1. Utilities НЕ викликають моделі.
2. Utilities НЕ будують графи.
3. Utilities НЕ повинні залежати від конкретного пайплайну.
4. Усі операції вводу-виводу — через Utilities.
5. Utilities не містять жодних бізнес-правил, тільки технічні операції.
6. Utilities мають бути максимально стабільними і не змінюватися при розширенні системи.

---

## 6.4. Взаємодія з іншими шарами

### Використовується:
- Data Adapters → file_utils, logger
- Graph Layer → graph_utils, graph_variations, file_utils
- Model Layer → metrics, logger, file_utils
- Pipelines → УСІ утиліти (як сервісний шар)

### Не повинні залежати:
- від моделей,
- від будівників графів,
- від пайплайнів,
- від адаптерів.

---

## 6.5. Вихідний результат

Після побудови Utilities Layer система отримає:

- єдиний центр для роботи з файлами,
- прозору систему логування,
- стандартизоване обчислення метрик,
- багату систему візуалізації,
- стабільні функції для роботи з графами,
- можливість розширення без переписування інших шарів.



# 7. High-Level Design (HLD): Configuration Layer

## 7.1. Призначення Configuration Layer

Configuration Layer забезпечує **централізоване, декларативне та відтворюване керування всіма параметрами системи**:

- джерела даних,
- побудова графів,
- вибір фіч,
- параметри моделей,
- гіперпараметри навчання,
- режими роботи (logs/bpmn/pokg),
- seed для експериментів,
- шляхи збереження результатів,
- параметри візуалізації,
- активація/деактивація модулів.

Layer є ключовим для **повторюваності експериментів**, порівняння моделей і ізоляції бізнес-процесів.

---

## 7.2. Основні задачі Configuration Layer

1. **Винос конфігурації за межі коду**  
   Жодні параметри не мають бути захардкожені.

2. **Підтримка набору конфігів на процес та модель**  
   ```
   config/
     processes/<process_id>/
       features.yaml
       model.yaml
       training.yaml
       graph.yaml
   ```

3. **Підтримка експериментальних конфігів**  
   Для статей, порівнянь та префіксних тестів.

4. **Забезпечення детермінізму**  
   seed, batch order, shuffle, graph normalization.

5. **Завантаження конфігів у будь-якій частині системи**  
   але тільки через центральний Config Manager.

---

## 7.3. Типи конфігів

### 7.3.1. data_sources.yaml

Описує всі джерела даних:
```
mssql:
  host: ...
  database: ...
  timeout: ...
xes:
  path: ...
rest:
  endpoint: ...
```

---

### 7.3.2. graph.yaml

Конфігурує Graph Construction Layer:
```
prefix:
  enabled: true
  max_length: 20
  strategy: "count"   # count/time/dynamic

features:
  time2vec: true
  global_stats: true
  doc_features: true
  bpmn_features: true
```

---

### 7.3.3. features.yaml

Керує формуванням фіч:
```
node_features:
  - duration
  - overdue_work
  - resource_id
  - role_id

edge_features:
  - time_diff
  - structural_relation

global_features:
  - prefix_length
  - num_unique_activities
```

---

### 7.3.4. model.yaml

Параметри моделей:
```
model_type: "GAT"
hidden_dim: 128
num_layers: 3
dropout: 0.3
heads: 4
time_encoding: "time2vec"
task:
  activity: true
  time: true
optimizer:
  type: "adamw"
  lr: 0.001
```

---

### 7.3.5. training.yaml

Параметри тренування:
```
epochs: 60
batch_size: 32
split:
  train: 0.7
  val: 0.15
  test: 0.15
seed: 42
early_stopping:
  patience: 8
  min_delta: 1e-4
```

---

### 7.3.6. modes.yaml

Описи режимів:
```
mode: "bpmn"       # logs / bpmn / pokg
enable_clustering: false
enable_variations: true
```

---

## 7.4. Config Manager

Це єдиний спосіб отримання конфігів:

```python
class ConfigManager:
    def __init__(self, process_id, model_id, mode):
        self.load_yaml_files(...)

    def get(self, section, key=None):
        pass
```

Переваги:
- централізація,
- кешування,
- контроль версій,
- логування завантажених конфігів.

---

## 7.5. Правила використання конфігів

1. Пайплайни **ніколи не хардкодять параметри**.
2. Моделі **не повинні містити значень параметрів усередині коду**.
3. Graph Builder читає тільки `graph.yaml` і `features.yaml`.
4. Model Layer читає тільки `model.yaml`.
5. Training Pipeline читає `training.yaml`.
6. Візуалізація читає `visualization.yaml` (якщо є).
7. Усі конфіги зберігаються разом з чекпоінтами:
   ```
   output/.../config_used/
   ```

---

## 7.6. Seed Management

Seed визначається в training.yaml і передається в:
- numpy,
- torch,
- random,
- graph normalization,
- shuffle алгоритми.

Важливо:
- seed задається на початку пайплайну,
- seed зберігається в метаданих експерименту.

---

## 7.7. Результат впровадження Configuration Layer

Система отримує:

- Повну керованість та повторюваність експериментів.
- Гнучке переключення режимів без переписування коду.
- Чисту, масштабовану архітектуру без хардкоду.
- Можливість автоматичного генератора експериментів.
- Ізоляцію моделей бізнес-процесів (model-per-process).




# 8. High-Level Design (HLD): Integration & Execution Flow

## 8.1. Призначення розділу

Цей розділ описує **повну інтеграцію всіх шарів системи** та **потоки виконання** у різних режимах:
- підготовка даних,
- побудова графів,
- тренування моделей,
- тестування та інференс,
- наукові експерименти.

Це — загальна схема роботи `bpm_prediction`, яку виконують користувачі, пайплайни, моделі та інструменти візуалізації.

---

## 8.2. Центральна концепція інтеграції

Система побудована так, що **кожен шар взаємодіє лише з сусідніми**:

```
User/CLI
   ↓
Pipelines Layer
   ↓
Data Adapters → Graph Construction → Feature Builder → Model Layer
   ↓
Utilities (логи, файли, діаграми)
   ↓
Output (моделі, графіки, статистика)
```

Немає циклічних залежностей.  
Немає прямого доступу моделей до адаптерів.  
Немає доступу графових модулів до пайплайнів.

---

## 8.3. Основні сценарії запуску

### 8.3.1. Сценарій 1 — Preprocessing

```
CLI → preprocess_pipeline
    → DataAdapter.load_events()
    → DataAdapter.load_bpmn()
    → file_utils.save_to_parquet()
```

Результат:
- сирі події,
- BPMN,
- проміжні таблиці,
- попередні статистики.

---

### 8.3.2. Сценарій 2 — Graph Construction

```
preprocess_pipeline
    → InstanceGraphBuilder.build(events)
    → BPMNGraphBuilder.build(bpmn)
    → FusionGraphBuilder.build(ig, bpmn, pokg)
    → FeatureBuilder.prepare(graph)
    → file_utils.save_graph()
```

Результат:
- IG, BPMN, Fusion Graph,
- підготовлені graph data objects,
- prefix-cut графи (за потреби).

---

### 8.3.3. Сценарій 3 — Training

```
train_pipeline
    → load prepared graphs
    → ConfigManager.load(model.yaml)
    → ModelFactory.create(model_type)
    → model.prepare_data(...)
    → model.train(...)
    → visualizer.save_training_diagram()
    → file_utils.save_checkpoint()
```

Результат:
- best model,
- last checkpoint,
- learning curves,
- зведені метрики.

---

### 8.3.4. Сценарій 4 — Evaluation (Test)

```
evaluate_pipeline
    → load graphs + labels
    → model.evaluate()
    → metrics.calculate_*
    → visualizer.plot_confusion_matrix()
    → visualizer.plot_prefix_metrics()
    → file_utils.save_statistics()
```

Результат:
- confusion matrices,
- accuracy/F1/AUPRC,
- prefix-length charts,
- BPMN vs Logs статистика,
- aggregated metrics.

---

### 8.3.5. Сценарій 5 — Inference (Production/PoC)

```
inference_server / CLI
    → DataAdapter.load_events(recent activity)
    → InstanceGraphBuilder.build(prefix)
    → FusionGraphBuilder.build(prefix, BPMN)
    → model.predict(graph)
```

Результат:
- прогноз наступної активності,
- прогноз часу,
- anomaly score.

---

### 8.3.6. Сценарій 6 — Науковий експеримент

```
experiment.yaml
    → batch_train(models=[GCN,GAT,TGAT,...], modes=[logs,bpmn,pokg])
    → batch_evaluate()
    → generate_prefix_bins()
    → compare_architectures()
    → t-tests()
    → visualizer.plot_regressions()
    → export to Excel/JSON
```

Результат:
- матеріали для статей (Scopus, конференції).

---

## 8.4. Потік даних (Data Flow Diagram)

```
             +-----------------------------+
             |           CLI (main.py)     |
             +-------------+---------------+
                           |
                           v
                +----------+----------+
                |       Pipelines     |
                +----------+----------+
                           |
      +--------------------+---------------------+
      |                                          |
      v                                          v
+-----+--------+                    +------------+-----------------+
|   Data Adapter |                  |       Graph Construction     |
| (MSSQL/XES/API)|                  |  IG / BPMN / Fusion Builders |
+------+--------+                   +------+------------------------+
       |                                    |
       v                                    v
+------+--------------------+       +-------+------------------------+
|      Feature Builder      |       |        Model Layer             |
|  node/edge/global/doc/... |       | GNN / TGNN / Transformer / AE |
+------+--------------------+       +-------+------------------------+
       |                                    |
       +---------------------------+---------+
                                   v
                     +-------------+----------------+
                     |            Utilities         |
                     | logging, file I/O, diagrams  |
                     +-------------+----------------+
                                   |
                                   v
                       +-----------+-------------+
                       |          Output         |
                       | models, plots, metrics  |
                       +-------------------------+
```

---

## 8.5. Інтеграція конфігурацій (Configuration Flow)

```
ConfigManager
    ↳ data_sources.yaml     → DataAdapter
    ↳ graph.yaml            → GraphBuilder
    ↳ features.yaml         → FeatureBuilder
    ↳ model.yaml            → Model Layer
    ↳ training.yaml         → Training Pipeline
    ↳ modes.yaml            → режим logs/bpmn/pokg
```

Усі конфіги зберігаються з експериментом.

---

## 8.6. Seed Flow (Reproducibility Contract)

```
CLI → seed
     ↳ random
     ↳ numpy
     ↳ torch
     ↳ graph normalization
     ↳ shuffling
```

Seed передається на рівні pipeline, а не моделей.

---

## 8.7. Error Handling Flow

Усі помилки проходять через `logger`:

- data source errors,
- graph inconsistencies,
- invalid feature definitions,
- missing model configs,
- training collapse,
- NaN на етапі loss.

Пайплайни зобов’язані обробляти помилки без падіння системи (idempotent).

---

## 8.8. Взаємодія з файловою системою

Пайплайни керують директоріями:

```
output/
  <process>/
    <model>/
      <mode>/
        <seed>/
          checkpoints/
          graphs/
          diagrams/
          metrics/
          confusion_matrices/
          prefix_stats/
          config_used/
```

Utilities забезпечують:
- читання/запис parquet, hdf5, csv,
- збереження графів,
- збереження checkpoint,
- агрегацію статистик.

---

## 8.9. Результат реалізації Integration & Execution Flow

Система отримає:

- повний контроль над усіма потоками,
- стандартизовані сценарії виконання,
- детермінованість усіх етапів,
- можливість легко додавати нові сценарії,
- підтримку одночасно наукових і production-процесів,
- автоматичне формування матеріалів для публікацій,
- стабільний та відтворюваний життєвий цикл моделей.





# 9. High-Level Design (HLD): Experimental Framework

## 9.1. Призначення Experimental Framework

Experimental Framework — це шар, який забезпечує:
- **повторювані, масштабовані та керовані експерименти**,  
- **порівняння архітектур**,  
- **статистичний аналіз** результатів,  
- **автоматичну генерацію матеріалів для наукових статей**,  
- **інтеграцію відкритих датасетів з власними** (Camunda/MS SQL + XES).

Цей шар перетворює систему `bpm_prediction` на професійний дослідницький інструмент.

---

## 9.2. Основні цілі Experimental Framework

1. **Підтримка пакетних експериментів (batch experiments):**
   - десятки моделей (GAT, TGAT, MuseGNN, Graphormer…)
   - сотні конфігів
   - декілька режимів (logs/bpmn/pokg)
   - множинні seed-и

2. **Порівняння моделей у режимах:**
   - Logs-only  
   - BPMN-aware  
   - BPMN + POKG  
   - (future) Logs + OrgStructureGraph

3. **Автоматичне вимірювання:**
   - accuracy / F1 / Top-3,
   - MAE/RMSE,
   - ROC-AUC / AUPRC,
   - ADR/FAR/FNR,
   - prefix-length impact,
   - per-activity confusion rates,
   - OOS performance.

4. **Автоматична генерація графіків:**
   - learning curves,
   - prefix charts,
   - regression curves,
   - aggregated confusion matrices,
   - radar charts порівняння моделей,
   - Logs vs BPMN vs POKG heatmaps.

5. **Статистичні тести:**
   - paired t-test (Logs vs BPMN),
   - cross-architecture comparisons,
   - distribution tests,
   - у майбутньому — Mann–Whitney та bootstrap CI.

6. **Автоматична збірка матеріалів для статей:**
   - зведення таблиць метрик,
   - рейтинг моделей,
   - графічні матеріали,
   - prefix bins,
   - статистично значущі відмінності.

---

## 9.3. Основні компоненти Experimental Framework

### 9.3.1. ExperimentRunner
Відповідає за:
- читання experiment.yaml,
- паралельний запуск моделей,
- агрегацію метрик,
- логування результатів.

Формат:

```
models:
  - GAT
  - TGAT
  - MuseGNN
modes:
  - logs
  - bpmn
  - pokg
prefix:
  enabled: true
  lengths: [1, 3, 5, 10, 20]
seeds: [1, 2, 3, 4, 5]
```

---

### 9.3.2. Experiment Registry

Усі експерименти зберігаються у структурі:

```
experiments/
  <experiment_name>/
    config_used/
    results.json
    metrics.csv
    diagrams/
    confusion/
    prefix_stats/
    logs/
```

Це дозволяє легко повторювати експерименти.

---

### 9.3.3. Batch Evaluator

Автоматичний аналіз:

- середні метрики по seed,
- стандартні відхилення,
- top-k моделей,
- таблиці порівнянь.

Використовує:
- `summarize_architecture_metrics`
- `aggregate_prefix_statistics`

---

### 9.3.4. Statistical Analysis Engine

Використовує:
- `paired_ttest_bpmn_vs_logs`
- `summarize_prefix_statistics`
- `visualize_diff_conf_matrix`

Аналізує:
- BPMN-aware improvements,
- prefix-length degradation,
- significance of model differences.

---

### 9.3.5. Visualization Engine

Забезпечує побудову:
- regression curves,
- prefix-length lines,
- radar charts,
- confusion matrices,
- aggregated matrices,
- bar charts,
- polynomial fits,
- архітектурних діаграм.

Це ключовий інструмент для створення графіків, необхідних для публікацій.

---

### 9.3.6. Dataset Integrator

Інструмент для об’єднання:
- локальних Camunda/MSSQL процесів,
- XES відкритих процесів,
- BPMN моделей з Camunda,
- BPMN моделей з відкритих датасетів.

Забезпечує:
- конвертацію XES → IG,
- уніфікацію атрибутів,
- створення єдиного POKG.

---

## 9.4. Потік роботи експерименту

```
experiment.yaml
     ↓
ExperimentRunner
     ↓
Pipelines (training + evaluation)
     ↓
Batch Evaluator
     ↓
Statistical Analysis Engine
     ↓
Visualization Engine
     ↓
results / diagrams / prefix_stats / tables
```

---

## 9.5. Підтримка префіксних експериментів

Experimental Framework автоматизує:
- генерацію графів від різних префіксів,
- окремий inference на кожному префіксі,
- порівняння prefix-bins між моделями,
- підготовку таблиць "prefix length → accuracy".

---

## 9.6. Підтримка статей (Scopus-ready workflow)

Фреймворк дозволяє автоматично зібрати матеріали для статті:

- загальні таблиці архітектур,
- середні метрики,
- статистично значущі відмінності,
- heatmaps логів проти BPMN,
- prefix charts,
- confusion matrices,
- порівняльні діаграми моделей.

Готові матеріали експортуються у:

```
export/
  tables/
  charts/
  diagrams/
  latex_ready/
  docx_ready/
```

---

## 9.7. Інваріанти Experimental Framework

1. Експерименти повинні бути повністю відтворюваними.
2. Жоден експеримент не може змінювати код моделі.
3. Експеримент лише комбінує конфіги.
4. Виконання експерименту не повинно падати на будь-якій моделі.
5. Усі метрики та діаграми повинні зберігатися структуровано.
6. Усі конфіги повинні зберігатись разом з результатами.

---

## 9.8. Результат впровадження Experimental Framework

Після реалізації:
- система дозволить автоматично порівнювати всі архітектури,
- експерименти будуть повністю відтворюваними,
- створення матеріалів для статей стане автоматизованим,
- відкриті та корпоративні процеси можуть порівнюватися,
- моделі можна тестувати на різних джерелах,
- експериментальні результати стануть якісно новими (рівень Scopus).




# 10. High-Level Design (HLD): Project Structure & Folder Layout

## 10.1. Мета розділу

Цей розділ визначає **офіційну структуру проєкту `bpm_prediction`**, яка забезпечує:

- відповідність Clean Architecture,
- чіткий поділ логіки за шарами,
- зручність для Codex/IDE,
- масштабованість,
- ізоляцію моделей для різних бізнес-процесів,
- правильне розміщення конфігів, графів, моделей, експериментів та артефактів.

Це фінальний шар HLD, який закріплює фізичну організацію директорій.

---

## 10.2. Загальна структура репозиторію

```
bpm_prediction/
│
├── main.py
├── requirements.txt
├── README.md
│
├── config/
│   ├── global/
│   │   ├── data_sources.yaml
│   │   ├── defaults.yaml
│   │   ├── modes.yaml
│   │   └── logging.yaml
│   │
│   └── processes/
│       └── <process_id>/
│           ├── features.yaml
│           ├── graph.yaml
│           ├── model.yaml
│           └── training.yaml
│
├── src/
│   ├── adapters/
│   │   ├── base_adapter.py
│   │   ├── mssql_adapter.py
│   │   ├── xes_adapter.py
│   │   ├── rest_adapter.py
│   │   └── mock_adapter.py
│   │
│   ├── graph/
│   │   ├── instance_graph_builder.py
│   │   ├── bpmn_graph_builder.py
│   │   ├── fusion_graph_builder.py
│   │   ├── feature_builder.py
│   │   └── graph_types.py
│   │
│   ├── models/
│   │   ├── base_model.py
│   │   ├── gnn/
│   │   │   ├── gcn.py
│   │   │   ├── gat.py
│   │   │   ├── graphsage.py
│   │   │   ├── mixhop.py
│   │   │   ├── gg
│   │   │   ├── tgat.py
│   │   │   ├── graphormer.py
│   │   │   └── ...
│   │   ├── transformer/
│   │   │   ├── encoder.py
│   │   │   └── hybrid.py
│   │   ├── classic/
│   │   │   ├── mlp.py
│   │   │   ├── rnn.py
│   │   │   └── autoencoder.py
│   │   └── model_factory.py
│   │
│   ├── pipelines/
│   │   ├── preprocess_pipeline.py
│   │   ├── train_pipeline.py
│   │   ├── train_predict_pipeline.py
│   │   ├── evaluate_pipeline.py
│   │   ├── generate_variations_pipeline.py
│   │   ├── doc_analysis_pipeline.py
│   │   ├── clustering_pipeline.py
│   │   └── experiment_runner.py
│   │
│   ├── utils/
│   │   ├── file_utils.py
│   │   ├── file_utils_l.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── graph_utils.py
│   │   ├── graph_creator.py
│   │   ├── graph_variations.py
│   │   ├── diagrams.py
│   │   ├── visualizer.py
│   │   └── string_utils.py
│   │
│   └── execution/
│       ├── config_manager.py
│       ├── seed_manager.py
│       ├── path_manager.py
│       └── error_handler.py
│
├── experiments/
│   ├── <experiment_name>/
│   └── experiment.yaml
│
├── data/
│   ├── raw/
│   ├── cached/
│   ├── graphs/
│   └── processed/
│
├── output/
│   ├── <process>/
│   │   ├── <model>/
│   │   │   ├── logs/
│   │   │   ├── checkpoints/
│   │   │   ├── diagrams/
│   │   │   ├── prefix_stats/
│   │   │   ├── confusion/
│   │   │   └── config_used/
│
└── tests/
    ├── test_graphs.py
    ├── test_models.py
    ├── test_pipelines.py
    ├── test_utils.py
    └── test_adapters.py
```

---

## 10.3. Принципи організації структури

### 10.3.1. Чітке розділення за шарами
- adapters → лише робота з даними,
- graph → лише побудова графів,
- models → лише ML,
- pipelines → orchestrations,
- utils → універсальні утиліти.

### 10.3.2. Zero-crossing dependency rule
Жоден модуль не має "стрибати" через шари та використовувати заборонені залежності.

### 10.3.3. Process-isolation rule
Для кожного бізнес-процесу своя папка конфігів + вихідних моделей.

### 10.3.4. Codex-ready structure
Кожен модуль містить мінімальні чіткі залежності → Codex легко аналізує.

---

## 10.4. Фізична структура віртуального середовища

Підтримуються дві стратегії запуску:

### 1. *Local Developer Mode* (на робочому ПК)
- глибока інтеграція з Pycharm,
- доступ до Codex-агента,
- доброякісне середовище для генерації коду.

### 2. *VM/Server Mode* (без Git доступу)
- read-only структура,
- запуски пайплайнів,
- експерименти,
- inference.

---

## 10.5. Правила розміщення файлів

### 10.5.1. Сорс-код
Тільки в `src/`.

### 10.5.2. Вхідні/вихідні дані
Тільки в `data/` та `output/`.

### 10.5.3. Проміжні кеші
Тільки в `data/cached/`.

### 10.5.4. Конфіги
Тільки в `config/`.

### 10.5.5. Скрипти міграції / утиліти
У корені або в папці `tools/` (опційно).

---

## 10.6. Результат впровадження

Після впровадження цього розділу:

- репозиторій стає **чистим, структурованим і придатним для командної розробки**,
- Codex може безпомилково працювати з цим проєктом,
- всі шари розділені фізично,
- проект легко переносити між середовищами,
- тестування стає ефективним,
- structure-based reasoning для ML-моделей працює коректно.

