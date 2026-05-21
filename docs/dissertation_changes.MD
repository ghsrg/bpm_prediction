# Dissertation Changes

Short checklist of project changes that should be reflected in the dissertation.

---

## 2026-05-21 - Training Layer Separation And Topology-Conditioned Methodology

- `topic`: separation of model fusion mechanism from training methodology
- `impact`: critical dissertation architecture clarification
- `status`: design direction
- `related_concepts`:
  - `training_layer=standard`
  - `training_layer=topology_conditioned`
  - `fusion_mode`
  - `versioned_zero_shot`
  - `controlled_forgetting`
  - `version_safe_mixed_negative_sampler`

**Alignment with current dissertation (ukr):**

Поточна дисертація вже містить рамку, у яку добре лягає новий підхід:
`Domain-Conditioned GNN`, ієрархічна стратегія адаптивного навчання,
`version-conditioned` регуляризація, `Zero-Shot` / `Few-Shot` адаптація,
структурний дрейф, контроль катастрофічного забування та предиктивне
управління в умовах еволюції бізнес-процесу. Тому `training_layer` не є
радикально новою темою, а є конкретизацією дисертаційної тези про
методологію навчання.

**Required dissertation clarification (ukr):**

Потрібно явно розділити три рівні:

```text
fusion_mode = як topology входить у forward model
training_layer = як організовано навчання, loss-и, sampling, negatives, retention
experiment.mode = який життєвий або оцінювальний сценарій запускається
```

Це важливо, бо поточні `fusion_mode` експерименти показують обмеження простої
подачі структури. Дисертаційний внесок має формулюватись не як "ще один спосіб
fusion", а як методологія навчання, яка змушує модель трактувати topology як
умову прогнозу:

```text
P(next_activity | observed_prefix, topology_version)
```

**Training layer concept (ukr):**

`training_layer=standard` має означати поточну поведінку навчання без зміни
результатів. Це контрольний режим.

`training_layer=topology_conditioned` має означати нову методологію навчання,
яка може застосовуватись до різних `fusion_mode`, але найбільший сенс має для
режимів, де topology впливає на hidden states або class ranking, зокрема
`StructXAttn` / майбутній `TopologyConditionedEncoder`.

**Topology-conditioned loss design (ukr):**

У тексті дисертації треба передбачити багатокомпонентну функцію втрат:

```text
L_total =
  L_ce_correct
  + lambda_wrong * L_wrong_version_negative
  + lambda_corrupt * L_edge_corruption_negative
  + lambda_allowed * L_allowed_set
  + lambda_retention * L_retention
```

Інтерпретація:

- `L_ce_correct`: базове навчання на правильній topology;
- `L_wrong_version_negative`: штраф, якщо модель не відрізняє правильну topology
  від topology іншої вже відомої версії;
- `L_edge_corruption_negative`: штраф, якщо модель ігнорує локально зіпсовану
  topology тієї самої версії;
- `L_allowed_set`: train-time узгодження прогнозу з allowed topology set, не те
  саме, що inference-time `experiment.mask_guided_enabled`;
- `L_retention`: контрольоване збереження якості на ще актуальних старих
  версіях.

**Zero-shot leakage rule (ukr):**

Для dissertation-grade `versioned_zero_shot` не можна використовувати future
topology як negative topology у фазі, де ця topology ще не відома бізнесу.
Наприклад, у фазі `train=v1` topology `v2/v3/v4` не можна використовувати навіть
як negative. Якщо доступна тільки одна версія, negative signal має будуватись
через `edge_corrupted_same_version`.

**Controlled forgetting (ukr):**

Катастрофічне забування треба замінити на поняття `controlled forgetting`.
У бізнес-сценарії старі версії ще деякий час працюють після появи нової, але з
часом втрачають актуальність. Тому retention має бути м'яким і залежним від
віку версії:

```text
recent versions -> higher retention weight
old versions -> lower retention weight
obsolete versions -> may be forgotten
```

Це краще відповідає реальному BPM lifecycle, ніж вимога зберігати однакову
якість на всіх історичних версіях.

**Versioned zero-shot protocol (ukr):**

`versioned_zero_shot` треба описати як автоматизований бізнес-життєвий цикл:

```text
Phase 1: train v1 -> eval v1 seen, v2/v3/v4 zero-shot
Phase 2: train/adapt v1+v2 -> eval v1/v2 seen, v3/v4 zero-shot
Phase 3: train/adapt v1+v2+v3 -> eval v1/v2/v3 seen, v4 zero-shot
```

Це не нова архітектура моделі, а оцінювальний та навчальний протокол, який
дозволяє відокремити:

- same-version quality;
- zero-shot benefit from new topology before logs are available;
- few-shot/adaptation benefit after logs of the new version accumulate;
- forgetting/degradation on older versions.

**Out-of-current-text changes (ukr):**

Ці уточнення частково виходять за поточний текст дисертації і потребують
оновлення розділів про методологію та архітектуру:

1. Додати явний термін `training_layer` або його український еквівалент як
   окремий рівень методології між архітектурою моделі та сценарієм експерименту.
2. Переформулювати "version-conditioned регуляризацію" як конкретний механізм:
   correct topology vs wrong/corrupted topology.
3. Додати leakage rule: future topology не використовується у навчанні або
   negative sampling до моменту появи відповідної версії в бізнес-сценарії.
4. Замінити жорстке формулювання про нейтралізацію catastrophic forgetting на
   більш бізнес-коректне `controlled forgetting`.
5. Окремо розвести `mask-guided constraint effect` і `topology-conditioned
   learning effect`.

**Do not forget:**

- у статті/дисертації не змішувати ефект `fusion_mode` та ефект
  `training_layer`;
- порівнювати мінімум `standard` vs `topology_conditioned` для одного й того
  самого `fusion_mode`;
- окремо показати матрицю:
  `standard/no-mask`, `standard/mask`, `topology_conditioned/no-mask`,
  `topology_conditioned/mask`;
- future work / technical debt: розширити negative sampler до повного
  `version_safe_mixed`, якщо перша реалізація з `wrong_version_topology` і
  `edge_corrupted_same_version` покаже корисний topology dependency signal.

---

## 2026-05-21 - Main Research Hypothesis: Topology-Conditioned Zero-Shot Drift Adaptation

- `topic`: topology-conditioned predictive monitoring under structural process drift
- `impact`: central dissertation/article hypothesis
- `status`: active research framing
- `dissertation_theme`: methodology of learning for optimizing business process management
- `business_assumption`: new BPMN/EOPKG topology is often available before sufficient logs of the new version exist
- `target_runtime_direction`: one dedicated fusion mode and training methodology, not a mixture of multiple fusion modes in one model

**Core hypothesis (ukr):**

Я бачу старі версії процесу та їхні логи. Модель має навчитися розуміти, як
topology впливає на `next-activity prediction`. Коли приходить нова topology
`vN`, навіть без логів `vN` для навчання, модель має використати її як умову
прогнозу.

**Why this matters (ukr):**

Це формулює бізнесову користь структури чіткіше, ніж звичайне порівняння
`baseline vs structure-aware GNN`. У реальних BPM-системах зміна процесу часто
спочатку з'являється як нова схема / BPMN / регламент, а журнали подій нової
версії накопичуються пізніше. Тому структура має бути корисною саме в
cold-start або zero-shot сценарії нової версії: модель ще не навчалась на логах
`vN`, але вже отримує topology `vN` як умову inference.

**Article framing (ukr):**

У статті можна показати:

1. `Baseline`: модель без структури не має доступу до нової topology і при
   structural drift покладається лише на патерни старих логів.
2. `Fusion-mode ablations`: різні поточні способи подачі структури
   (`ClassMeanAttention`, `ClassMeanConcat`, `ClassAwareStructuralScoring`,
   `TopologyStateEncoder`, `TopologyStateGraphEncoder`, `StructuralPriorEncoder`,
   `StructXAttn`) дають часткові покращення або покращують OOS/ECE/process
   consistency, але не дають стабільної статистично значущої переваги для
   drift strict-F1.
3. `Methodology gap`: це не спростовує користь topology, а показує, що проста
   подача структури недостатня. Потрібна спеціальна methodology навчання, яка
   змушує одну topology-conditioned модель реально залежати від topology.
4. `Perspective / proposed direction`: новий єдиний fusion режим на кшталт
   `TopologyConditionedEncoder`, навчений через correct-vs-corrupted topology,
   allowed-set loss, OOS penalty, version/structure augmentation і
   transfer-validation.

**Important constraint (ukr):**

Не робити основною ідеєю змішування двох різних `fusion_mode` в одній моделі
(`TopologyStateEncoder teacher + StructXAttn student` тощо). Це може бути
допоміжний експеримент, але для дисертації та Scopus статті краще мати одну
чітку модельну концепцію: topology-conditioned predictive monitoring.

**Do not forget:**

- формулювати внесок як methodology of learning, а не як черговий варіант
  fusion;
- окремо підкреслити zero-shot/cold-start цінність: topology нової версії
  доступна до появи достатніх логів;
- baseline має бути принципово слабшим саме тому, що не бачить topology `vN`;
- поточні fusion experiments показують motivation: є signal, але без правильної
  training methodology він не переноситься стабільно;
- головні метрики для доказу: drift-window strict/non-strict F1, OOS,
  `target_in_mask_rate`, `pred_in_mask_rate`, ECE/NLL, seed variance,
  leave-one-version-out або train-old/eval-new protocol.

---

## 2026-05-17 - Fusion Architecture Comparison And StructXAttn Direction

- `topic`: dual-stream structure-aware learning under structural drift
- `impact`: critical for methodology and future work
- `status`: research conclusion / next architecture candidate
- `related_runtime`:
  - `ClassMeanAttention`
  - `ClassMeanConcat`
  - `ClassAwareStructuralScoring`
  - `TopologyStateEncoder`
  - `TopologyStateGraphEncoder`
  - `StructuralPriorEncoder`
  - `StructXAttn` / `LayerwiseStructuralCrossAttention` as proposed next mode

**Note (ukr):**

Поточні режими `EOPKGGATv2` треба описувати як послідовність ablation-рівнів
інтеграції структури, а не як один фінальний підхід до OOD-generalization.
Експерименти показали, що проста подача BPMN/EOPKG структури може зменшувати
OOS, ECE та інколи покращувати non-strict / hybrid метрики, але поки не дає
стабільного приросту drift strict-F1. Це важливий методологічний висновок:
структурний сигнал сам по собі корисний, але спосіб його введення та стратегія
навчання визначають, чи буде він переноситись між версіями процесу.

**Architecture comparison (ukr):**

| Architecture | Strong side | Weak side for drift transfer |
|---|---|---|
| `ClassMeanAttention` | Зберігає observed encoder і додає structural attention до representation. | Один `obs_context` стискає весь prefix у глобальний query; topology signal проходить через bottleneck і не модулює IG message passing. |
| `ClassMeanConcat` | Простий контрольний baseline для representation-level structural context. | Mean-pooled структура згладжує топологічні відмінності; weak signal для strict class ranking. |
| `StructuralPriorEncoder` | Etalon-like observed + pooled structural prior; добре підходить як стабільний representation-level baseline. | Структура лишається global prior після observed encoder; не змушує IG encoder вчитись залежати від поточної топології. |
| `ClassAwareStructuralScoring` | Структура прямо додає per-class structural logits; добре показує вплив на ranking, OOS/ECE і mask-compatible diagnostics. | Це late fusion: structural branch коригує logits, але не змінює hidden states observed/IG encoder. |
| `TopologyStateEncoder` | Перевіряє early topology-state signal і node-to-class structural prediction. | Observed IG encoder bypassed; може втрачати sequential/instance dynamics і давати structural overfitting. |
| `TopologyStateGraphEncoder` | Найближчий до старого `etalon`: topology graph стає власним prediction graph. | Також bypassed observed IG encoder; hard `structural_edge_index` може погано переноситись на unseen topology. |
| `StructXAttn` / `LayerwiseStructuralCrossAttention` | Потенційно правильний dual-stream механізм: IG node states формують `Q`, structural nodes формують `K,V`, структура впливає на hidden states до classifier. | Потребує нової training methodology, інакше модель може знову ігнорувати structural stream. |

**Main conclusion (ukr):**

Для дисертації важливо не стверджувати, що "будь-яка структура покращує
strict-F1 під drift". Коректніший висновок: поточні late/global/structural-only
режими демонструють обмеження простого fusion і мотивують перехід до
dual-stream token-level / layerwise structural cross-attention. Саме це
пояснює, чому наступним кроком має бути `StructXAttn`, де структура не тільки
штрафує фінальні logits, а бере участь у формуванні observed representation.

**StructXAttn target idea:**

```text
Q = W_Q h_IG_node
K = W_K h_struct_node
V = W_V h_struct_node
h_IG_node' = h_IG_node + gamma * CrossAttention(Q, K, V)
```

У цій постановці `structural_mode=true` означає не "додати pooled structure",
а "condition observed IG encoder on current process topology". Для transfer
між `v1 -> v2/v3/v4` цього все одно недостатньо без навчальної методології:
потрібні structural dropout, corrupted-topology contrastive checks,
multi-version або pseudo-version training, а також diagnostics, які доводять,
що модель реально використовує structural stream.

**Do not forget:**

- описати поточні fusion modes як ablation hierarchy: global context, class-aware
  logits, topology-state-only, structural graph baseline, future token-level
  cross-attention;
- для Scopus не робити сильний claim про strict-F1 benefit без multi-seed і
  transfer-validation;
- показувати OOS, ECE, NLL, hybrid/non-strict F1 окремо від strict-F1;
- додати future-work / next-methodology блок про `StructXAttn` і structural
  conditioning of IG encoder;
- додати контроль `correct topology vs corrupted topology` як доказ, що
  structural branch не декоративний.

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

---

## 2026-05-17 - StructXAttn / Token-Level Structural Cross-Attention

- `topic`: structural fusion and training methodology
- `impact`: critical for dissertation hypothesis refinement
- `status`: implemented experimental mode
- `related_runtime`: `model.fusion_mode=StructXAttn`

**Note (ukr):**

Додано `StructXAttn` як TokenXAttn-style dual-stream fusion: observed IG nodes
роблять cross-attention до structural nodes до pooling/classifier. Це важливо
для дисертації, бо попередні режими показали, що late/global structural fusion
не гарантує перенесення користі структури при structural drift. Нова гіпотеза
формулюється точніше: структура має керувати формуванням observed hidden states,
а модель має бути навчена не ігнорувати structural stream.

Реалізовано два placement режими:

1. `model.struct_xattn_layers=post_conv2` - безпечний smoke/integration режим;
2. `model.struct_xattn_layers=after_each_conv` - основний research режим, де
   structural guidance входить після кожного observed GATv2 message-passing
   шару.

Окремо додано training methodology:

```text
correct topology vs corrupted topology
```

Під час train модель порівнює CE на правильній структурі і CE на зіпсованій
структурі. Це має показати, чи correct topology справді корисна для моделі.
В eval/eval_drift corrupted forward не використовується.

**Do not forget:**

- у methodology пояснити, що `StructXAttn` відрізняється від
  `ClassMeanAttention`: query робить кожен observed node, а не один pooled
  `obs_context`;
- окремо розділяти архітектурний ефект `post_conv2` / `after_each_conv` і
  ефект contrastive topology training;
- при аналізі результатів дивитись не тільки F1, а й:
  `struct_xattn_to_observed_ratio`, `struct_xattn_attention_entropy`,
  `struct_xattn_correct_vs_corrupt_ce_delta`,
  `struct_xattn_prediction_flip_rate`;
- якщо `correct_vs_corrupt_ce_delta <= 0`, це означає, що модель все ще
  ігнорує структуру або corruption policy не створює змістовний negative view;
- це не фінальний доказ transfer між версіями, а наступний крок після висновку,
  що проста подача структури не дає стабільного strict-F1 ефекту.

---

## 2026-05-21 - Topology-Conditioned Learning Strategy

- `topic`: training methodology
- `impact`: critical for dissertation core hypothesis
- `status`: implemented trainer-level strategy
- `related_runtime`: `training.learning_strategy=topology_conditioned`

**Note (ukr):**

Додано відокремлення `fusion_mode` від `learning_strategy`. Це важливо для
дисертації: тепер можна окремо досліджувати, як структура входить у forward
моделі, і як модель спеціально навчається залежати від topology.

`topology_conditioned` не є новою архітектурою fusion. Це методологія навчання,
яка додає до базового CE:

1. `L_allowed_set` - train-time consistency з дозволеним topology target set;
2. `L_wrong_version_negative` - порівняння правильної topology з topology іншої
   вже відомої версії;
3. `L_drop_edges_negative` - контрольована same-version topology corruption;
4. `L_retention` - version-weighted CE/rehearsal для вже бачених версій.

**Do not forget:**

- у методології описати, що `mask_guided_enabled` і `L_allowed_set` - різні
  механізми: перший змінює logits policy, другий формує train-time loss;
- для Scopus/дисертації порівнювати матрицю:
  `standard/no-mask`, `standard/mask`, `topology_conditioned/no-mask`,
  `topology_conditioned/mask`;
- основні diagnostics:
  `topology_conditioned_wrong_version_ce_delta`,
  `topology_conditioned_drop_edges_ce_delta`,
  `topology_conditioned_allowed_set_loss`,
  `topology_conditioned_retention_loss`,
  `topology_conditioned_mixed_version_batches`;
- перша реалізація використовує physical `drop_edges`, бо structural GNN ще не
  споживає `structural_edge_weight`; Soft Edge Drop треба описувати як future
  work/tech debt.
