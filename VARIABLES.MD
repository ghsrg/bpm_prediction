Цей словник фіксує єдиний стандарт найменувань (`snake_case`) для синхронізації математичного апарату з програмним кодом (PyTorch/pm4py).

### 1. Структура графа та простору
* **$G_{obs}$** | `observed_graph` — Спостережуваний граф виконання (InstanceGraph), що представляє виконаний префікс процесу.
* **$G_{struct}$** | `structural_graph` — Нормативний граф (EPOKG), що відображає регламентовану структуру процесу. (Активується з MVP2)
* **$G_{fusion}$** | `fusion_graph` — Мультимодальний граф злиття, що поєднує виконання та нормативний контекст. (MVP2+)
* **$G^{(t)}$** | `process_graph` — Стан графа процесу (екземпляра або нормативної моделі) у момент часу $t$.
* **$V, E$** | `nodes`, `edges` — Множини вузлів (активностей) та ребер (зв'язків).
* **$A^{(t)}$** | `adj_matrix` — Матриця суміжності графа розмірності $|V| \times |V|$.

### 2. Версійність та латентний простір
* **$\kappa$** | `process_version` — Дискретний ідентифікатор версії регламенту (бізенс-процесу).
* **$e_\kappa$** | `version_emb` — Ембеддінг поточної версії регламенту у латентному просторі.
* **$\kappa \rightarrow \kappa+1$** | `version_transition_operator` — Оператор переходу між версіями процесу. (MVP4+)
* **$X_t$** | `node_features` — Матриця вхідних ознак вузлів.
* **$Z_t$** | `latent_repr` (Latent Representation) — Матриця латентних подань екземпляра після обробки GNN.
* **$z_{plan}^{(\kappa)}$** | `macro_plan_vec` — Вектор нормативного макро-плану поточної версії $\kappa$ (використовується модулем «Критик»).

## 3. Prefix & Latent Representation
* **$\sigma$** | `execution_prefix` — Виконаний префікс траси до моменту прогнозування.
* **$h_{\sigma}$** | `prefix_embedding` — Латентне представлення префіксу після GNN та global pooling.
* **$c_{\sigma}$** | `context_embedding` — Контекстна репрезентація (версія процесу κ або структурний контекст).
* **$z_{fused}$** | `fused_representation` — Результат оператора Γ, що об’єднує $h_{\sigma}$ та $c_{\sigma}$.

## 4. Masking & Constraints
* **$M_{struct}$** | `structural_mask` — Детермінована маска допустимих переходів. (MVP2+)
* **$s_{macro}$** | `macro_stability_state` — Глобальний стан стабільності системи (0 — стабільно, 1 — реконфігурація). (MVP3+)

### 5. Навчання та оптимізація, Training & Optimization
* **$\theta$** | `model_parameters` — Глобальні параметри (ваги) графової нейромережі.
* **$\Delta \theta$** | `parameter_update` — Оновлення параметрів під час навчання.
* **$t$** | `time_step` — Часовий крок або ітерація навчання.
* **$\mathcal{L}_{reg}$** | `loss_reg` — Функція структурного регуляризатора (штраф за відхилення від нормативного графа).
* **$L_{pred}$** | `prediction_loss` — Основна функція втрат (CrossEntropy).
* **$L_{critic}$** | `critic_loss` — Додатковий штраф від модуля Critic. (MVP3+)
* **$L_{drift}$** | `drift_loss` — Штраф за виявлений дрейф. (MVP4+)
* **$L_{total}$** | `total_loss` — Загальна функція втрат.
* **$\beta^{(t)}$** | `beta_relax` (Relaxation Coefficient) — Динамічний коефіцієнт релаксації, який алгоритмічно розслабляє структурний регуляризатор при детекції дрейфу.
* **$\tau_{ood}$** | `ood_threshold` — Специфічний поріг OOD для структурного дрейфу (по $\mathcal{W}_1$).
* **$R_{score}$** | `reliability_score` — Агрегований числовий показник ризику.
* **$S_{rel}$** | `reliability_state` — Дискретний стан (green/yellow/red).* **$\tau_{conf}$** | `confidence_threshold` — Порогове значення впевненості.

### 6. Оператори злиття контексту (з дисертації)
* **$\Gamma$** | `knowledge_infusion_operator` — Оператор ін'єкції знань, що нелінійно поєднує локальний стан екземпляра та глобальний контекст EPOKG.
  * **Кодова примітка:** у реалізації це має бути `nn.Module`/`nn.Sequential`, а не “просто вектор”.
  * **Кодова форма:** `fusion_repr = self.knowledge_infusion_operator(local_state_vec, global_context_vec)`.
* **$c_\sigma$** | `global_context_vec` — Вектор релевантного глобального контексту, екстрагований для поточного екземпляра.
* **$h_\sigma$** | `local_state_vec` — Локальне векторне подання поточного стану кейсу (trace).

### 7. Drift Signals

* **$D_{data}$** | `data_drift_score` — Індекс дрейфу атрибутивних розподілів.
* **$D_{concept}$** | `concept_drift_score` — Індекс концептуального дрейфу (feature→target).
* **$\delta_{drift}$** | `drift_signal` — Узагальнений сигнал дрейфу.
* **$\mathcal{W}_1$** | `wasserstein_distance` — Метрика Wasserstein-1 для оцінки структурного зсуву.
* **$OOD_{latent}$** | `latent_ood_flag` — Прапорець латентного OOD.
* **$OOD_{struct}$** | `structural_ood_flag` — Прапорець структурного OOD.

### 8. Reliability & Thresholds

* **$R_{score}$** | `reliability_score` — Агрегований скор ризику.
* **$\tau_{warn}$** | `warning_threshold` — Поріг переходу у режим "yellow".
* **$\tau_{crit}$** | `critical_threshold` — Поріг переходу у режим "red".
* **$\tau_{ood}$** | `ood_threshold` — Локальний поріг структурного OOD.

### 9. Adaptation & Control State

* **$S_{rel}$** | `reliability_state` — Дискретний стан семафора (`green` / `yellow` / `red`).
* **$n_{min}$** | `min_buffer_size` — Мінімальний розмір буфера для Few-Shot донавчання.

### 10. Ієрархія порогів (узгоджені правила)
1. `ood_threshold` ($\tau_{ood}$) — локальний поріг структурного OOD (для $\mathcal{W}_1$).
2. `warning_threshold` ($\tau_{warn}$) і `critical_threshold` ($\tau_{crit}$) — керуючі пороги Reliability Semaphore на агрегованому рівні.
3. Рекомендований інваріант: $\tau_{warn} < \tau_{crit}$.

## 11. Prediction
* **$\hat{y}$** | `predicted_activity` — Прогнозована наступна активність.
* **$y$** | `true_activity` — Фактична наступна активність.
* **$\text{logits}$** | `logits` — Нормалізовані значення перед softmax.
* **$\text{confidence}$** | `confidence_score` — Максимальна ймовірність після softmax.


### 12. Трекінг для експериментів (MLflow)
* Для режимів `inference` та `evaluate` логувати в трекер (MLflow):
  - `data_drift_score`
  - `concept_drift_score`
  - `wasserstein_drift`
  - `reliability_score`
  - `semaphore_mode`
* Мінімальна частота логування: кожний batch або інший атомарний крок інференсу.
