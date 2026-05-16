# Dissertation Changes

Short checklist of project changes that should be reflected in the dissertation.

---

## 2026-05-16 - TopologyStateGraphEncoder Structural Graph Baseline

- `topic`: etalon/article-like structural graph baseline
- `impact`: critical for methodology
- `status`: implemented experimental baseline
- `related_runtime`: `model.fusion_mode=TopologyStateGraphEncoder`

**Note (ukr):**

Додано окремий baseline, де BPMN/EOPKG topology є самим графом прогнозування:
стан префікса проектується на structural nodes, далі виконується structural
GNN message passing і graph-level readout. Це ближче до старого `etalon`, ніж
`StructuralPriorEncoder`, бо структура не стискається у mean context після
observed encoder, а бере участь у message passing до класифікації.

**Do not forget:**

- порівнювати `Base-UN`, `StructuralPriorEncoder`, `TopologyStateEncoder`,
  `TopologyStateGraphEncoder`, `ClassAwareStructuralScoring`;
- спочатку перевірити same-version користь структури, і лише після цього
  аналізувати transfer між версіями;
- не називати цей режим final drift solution: hard `structural_edge_index`
  може давати structural overfitting;
- показувати diagnostics `topology_graph_context_mean_abs`,
  `topology_graph_logits_mean_abs`, `topology_graph_entropy`;
- old graph caches without `struct_prefix_state_x` must be rebuilt before this
  mode can run.

---

## 2026-05-13 - Early TopologyStateEncoder Fusion Ablation

- `topic`: fusion level for structure-aware predictive monitoring
- `impact`: critical for methodology
- `status`: implemented experimental ablation
- `related_runtime`: `model.fusion_mode=TopologyStateEncoder`

**Note (ukr):**

Додано окремий early/input-level fusion режим `TopologyStateEncoder`. На
відміну від `ClassAwareStructuralScoring`, який додає structural logits після
observed encoder, цей режим переносить стан prefix безпосередньо на structural
nodes перед structural GNN:

```text
h0_v = LayerNorm(base_struct_v + beta * gate(prefix_state_v) * W_prefix(prefix_state_v))
class_logit_j = pool({W_node StructuralGNN(h0)_v | struct_node_to_class_index[v] = j})
```

Це треба описувати не як фінальне рішення drift generalization, а як ablation
для перевірки рівня злиття: чи є користь, коли observed prefix state входить у
структурний encoder раніше, до class logits. Важливо прямо зазначити ризик:
жорсткий `structural_edge_index` може створювати structural overfitting до
train topology і negative transfer при переході на нову версію процесу.

**Do not forget:**

- порівняти `ClassAwareStructuralScoring` vs `TopologyStateEncoder` як
  logit-level fusion vs early/input-level fusion;
- не робити висновок, що early fusion автоматично є SOTA рішенням для drift;
- показати diagnostics: `topology_state_entropy`, `topology_state_gate_mean`,
  `topology_state_gate_max`, `topology_state_mean_class_cardinality`,
  `topology_state_max_class_cardinality`;
- пояснити `logmeanexp` pooling як захист від inflation для duplicate
  activity nodes;
- якщо early fusion погіршує unseen versions, це підтримує тезу про потребу
  окремої methodology для структурного перенесення, а не спростовує користь
  BPMN/topology як джерела сигналу.

---

## 2026-05-07 - Class-Aware Structural Scoring Fusion

- `topic`: structure-aware GNN fusion semantics
- `impact`: critical
- `files`:
  - `src/domain/models/eopkg_models.py`
  - `src/application/use_cases/trainer.py`
  - `docs/GNN_RUNTIME_MVP2_5.MD`

**Note (ukr):**

У дисертації треба явно розділити старий підхід, де структура зводилась до
одного global structural context, і новий підхід
`ClassAwareStructuralScoring`, де структура формує окремий structural logit для
кожного candidate activity. Це критично для інтерпретації експериментів:
топологія тепер впливає безпосередньо на ranking класів, а не лише додає
середній контекст до observed branch.

**Formula reminder:**

```text
node_score(i, v) = q_i^T W_bilinear k_v + W_prior h_v
raw_struct_class(i, j) = LogSumExp({node_score(i, v) | class(v) = j})
structural_logits(i, :) = gamma * obs_scale(i) * LayerNorm(raw_struct_class(i, :))
final_logits = observed_logits + structural_logits
```

**Do not forget:**

- explain `struct_node_to_class_index` as the explicit `|V| -> C` projection;
- mention late pooling with `LogSumExp` for duplicated activity labels;
- report structural contribution diagnostics:
  `structural_logits_mean_abs`, `structural_logits_max_abs`,
  `observed_logits_mean_abs`, `structural_to_observed_logit_ratio`;
- describe auxiliary structural loss as a validation/control mechanism for
  teaching the structural branch, not as a replacement for the main CE loss;
- separate results for `ClassMeanAttention`, `ClassMeanConcat`, and
  `ClassAwareStructuralScoring`.

---

## 2026-05-08 - Additive Stats Enrichment for Structural Node Features

- `topic`: structure-aware GNN input semantics
- `impact`: critical
- `files`:
  - `src/domain/models/eopkg_models.py`
  - `docs/GNN_RUNTIME_MVP2_5.MD`

**Note (ukr):**

У дисертації треба підкреслити, що статистичні ознаки `struct_x` більше не
замінюють structural identity embedding. Вони додаються як контрольоване
збагачення перед structural GNN:

```text
h0_v = LayerNorm(identity_v + beta * stats_projection(struct_x_v))
```

Це важливо для інтерпретації експериментів: topology identity залишається
основним структурним носієм, а runtime/process statistics є додатковим
сигналом. Параметр `model.structural_stats_beta` треба описати як ablation
control для сили статистичного внеску.

**Do not forget:**

- порівняти `beta=0.0`, `0.1`, `0.3`, `1.0`;
- окремо інтерпретувати `structural_stats_beta` і `structural_logit_scale`:
  перший масштабує stats input до structural GNN, другий масштабує structural
  logits після class-aware scoring;
- у Scopus/дисертаційному тексті не називати це просто "statistics enabled",
  а формулювати як "additive statistical enrichment of topology identity".

---

## 2026-05-09 - Structural Signal Negative Transfer Under Drift

- `topic`: empirical validity of structure-aware drift generalization
- `impact`: critical
- `status`: preliminary finding
- `scenario`: Train v1, drift test v1-v4
- `runs_to_reference`:
  - `8a1c0991022a45ccb3e02d45c6c29de8` - aggressive structure/statistics run

**Note (ukr):**

Проміжний висновок: сама наявність BPMN/topology structure у GNN не гарантує
кращої узагальнюваності при structural drift. У сценарії `Train: v1 -> Drift
test: v1, v2, v3, v4` агресивно підсилена structural branch показала negative
transfer: при переході на `v2` baseline і помірна структура падали приблизно з
`0.85` до `0.55-0.60`, а підсилена структура падала сильніше - приблизно до
`0.49`. На `v3` і `v4` картина схожа або трохи гірша.

Це означає, що structural signal у поточному вигляді може вчитися як
version-specific prior для `v1`, а не як version-invariant process reasoning.
Сильне збільшення `structural_stats_beta`, `structural_logit_scale` і
`structural_aux_loss_weight` доводить, що структура бере участь у logits і loss,
але не доводить її корисність для drift generalization. Навпаки, занадто
сильний structural prior може погіршити перенесення на нову версію процесу.

Водночас це не означає, що структура "не працює" у всіх сенсах. Попередні
запуски показували, що structural conditioning може зменшувати OOS/invalid
behavior, покращувати узгодженість прогнозу з allowed topology/mask, підвищувати
`target_in_mask_rate` / `pred_in_mask_rate` або пов'язані confidence/calibration
показники. Тобто структура може робити модель більш process-consistent, навіть
якщо strict next-activity F1 на unseen версіях не зростає або погіршується.

**Interpretation for dissertation:**

- `structure-aware` не можна подавати як автоматичне покращення predictive
  monitoring under drift;
- потрібна окрема methodology для навчання, валідації та інтерпретації
  structural signal;
- треба розрізняти `structural participation` і `structural usefulness`:
  перше видно через `structural_to_observed_logit_ratio` та aux loss, друге -
  через стабільність strict-F1 на unseen versions;
- треба окремо розрізняти `strict predictive accuracy` і
  `process-consistency`: структура може покращувати OOS/mask/confidence
  behavior без приросту exact next-activity F1;
- агресивні structural parameters корисні як stress test, але не як основний
  research-grade режим;
- ключове питання для наступного етапу: як навчити structural branch не
  підганятися під одну train topology, а переносити стабільні закономірності
  між версіями процесу.

**Do not forget:**

- описати цей результат як мотивацію для validity conditions перед drift
  analysis;
- порівняти `Train v1 -> Test v1-v4` і `Train v1,v2 -> Test v1-v4`;
- окремо зробити baseline, moderate structure, aggressive structure,
  structure+stats, structure+stats+mask;
- у таблицях показувати не лише absolute F1, а й degradation від seen version:
  `F1(v1) - F1(vk)` або relative degradation;
- додати окремі таблиці/графіки для OOS, `target_in_mask_rate`,
  `pred_in_mask_rate`, confidence/calibration metrics і mask-cardinality groups;
- не робити висновок "структура не працює взагалі"; коректніше:
  "naive or over-amplified structural conditioning is insufficient and may
  introduce negative transfer under structural drift, while still improving
  process-consistency and out-of-structure behavior".

---

## 2026-05-12 - Runtime Validity for Stats-Backed Structural Drift

- `topic`: experiment validity and runtime reliability
- `impact`: important
- `status`: implemented runtime guard; full drift validation pending
- `related_debt`: `stats_backed_structural_payload_caching`

**Note (ukr):**

Для dissertation-grade drift experiments зі `structural_mode=true` та
`statistic_enabled=true` важливо зафіксувати не лише метрики моделі, а й умову
валідності runtime: stats snapshots і structural payloads не повинні
матеріалізуватись заново для кожного prefix graph. Інакше експеримент може
падати через Neo4j/driver/RAM pressure, а відсутність результату буде
інфраструктурним bias, не властивістю моделі.

Після зміни runtime використовує:

- lightweight Neo4j snapshot timeline lookup;
- heavy stats payload cache by resolved snapshot identity;
- deduplicated sharded structural payloads by `structural_payload_key`;
- diagnostics `GRAPH_DATASET_STRUCT_PAYLOADS`.

**Do not forget:**

- у методології згадати, що full drift experiments перевірялись на runtime
  feasibility та snapshot-aware stats caching;
- для фінальних запусків перевіряти, що `GRAPH_DATASET_STRUCT_PAYLOADS`
  показує кількість structural payloads значно меншу за кількість prefix graphs;
- не змішувати runtime validity з model quality: ця зміна не повинна сама по
  собі покращувати F1, вона прибирає інфраструктурне обмеження для коректного
  вимірювання.

---

## 2026-05-13 - One-Pass Drift Window Evaluation

- `topic`: experiment validity and drift runtime scalability
- `impact`: important
- `status`: implemented; full drift rerun pending
- `related_debt`: `stats_backed_structural_payload_caching`

**Note (ukr):**

Для dissertation-grade `eval_drift` важливо, щоб drift-window оцінка не
перебудовувала графи та статистично-збагачену структуру для кожного sliding
window. Інакше runtime overhead залежить від кількості вікон і може стати
інфраструктурним bias: експеримент не завершується або займає непропорційно
багато часу саме для structure+stats режимів.

Після зміни `eval_drift` може виконувати один inference pass по prebuilt
`test_dataset`, зберігати компактні per-sample records і агрегувати
drift-window metrics без повторного graph build та без повторного forward по
overlapping windows.

**Do not forget:**

- у методології згадати, що drift windows рахуються з фіксованого prebuilt
  graph dataset, а не через повторну побудову графів;
- для великих логів не зберігати повний `N x C` probability tensor; достатньо
  scalar records (`confidence`, `correct`, `top3_hit`, mask flags, set NLL);
- при повторенні експериментів перевіряти лог
  `Drift evaluation path: one_pass_prebuilt_dataset`;
- старі graph cache entries без `trace_idx` треба перебудувати, бо вони
  підуть у fallback `legacy_raw_trace_windows`.
---

## 2026-05-14 - StructuralPriorEncoder Baseline

- `topic`: structure as representation-level prior
- `impact`: important
- `status`: implemented baseline mode
- `related_runtime`: `model.fusion_mode=StructuralPriorEncoder`

**Note (ukr):**

Додано etalon-like baseline, у якому структура процесу використовується як
representation-level prior перед classifier, а не як mask, не як окремий
structural-logit scorer і не як topology-only predictor. Це важливо для
дисертації, бо дозволяє розділити два питання:

1. чи структура корисна для прогнозування наступної активності в межах
   узгодженої/відомої версії процесу;
2. чи переноситься ця користь між версіями процесу під structural drift.

Режим `structural_prior_fusion=concat` є найближчою аналогією до старого
`etalon`: observed graph context і structural graph context зливаються перед
classification head. Режим `gated_concat` є стабілізованою абляцією, де модель
навчається, скільки structural context пропускати.

**Do not forget:**

- у методології окремо назвати цей режим clean structural baseline;
- не змішувати його висновки з `ClassAwareStructuralScoring`, бо там структура
  впливає на ranking класів через logits;
- порівнювати `Base-UN`, `StructuralPriorEncoder/concat`,
  `StructuralPriorEncoder/gated_concat`, а вже потім drift transfer;
- при аналізі показувати `observed_context_mean_abs`,
  `structural_prior_context_mean_abs`,
  `structural_prior_to_observed_context_ratio`,
  `structural_prior_gate_mean`, `structural_prior_gate_max`.
