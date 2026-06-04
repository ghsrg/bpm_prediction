# Dissertation Changes

Short checklist of project changes that should be reflected in the dissertation.

---

## 2026-05-31 - EOPKGTopologyConditioned Impulse Activation and Topology State Routing

- `topic`: dynamic execution pointer conditioning and impulse activation in structural GNN
- `impact`: proposed hypothesis for aligning EOPKGTopologyConditioned with dissertation theoretical methodology; planned rework for investigating structural concept drift degradation in topology-native candidate space, not a validated solution
- `status`: proposed hypothesis / planned rework (awaiting review and empirical validation)
- `related_concepts`:
  - `impulse_activation_routing`
  - `struct_prefix_state_x` (`is_last_event`)
  - `inductance_message_passing`
  - `topology_state_routing`
  - `bilinear_candidate_matching`
  - `latent_flow_space` (Stage 2)

**Dissertation notes (ukr):**
- Попередні EOPKG fusion-підходи варто описати як ablation path: структура як late/static prior або як early topology-only branch не дала стабільного zero-shot перенесення.
- Нова гіпотеза: `EOPKGTopologyConditioned` має прогнозувати в topology-native candidate space (`candidate_logits [B, K_v]`), а не через fixed-label logits `[B, C_train]`.
- `struct_prefix_state_x.is_last_event` треба трактувати як execution pointer / impulse source, який активує поточну topology перед candidate scoring.
- `ObservedEncoder` має залишатися активним: topology routing не замінює sequential evidence, а умовлює candidate ranking поточною структурою.
- `latent_flow_space` для статистики переходів лишається наступним planned stage, не частиною вже доведеного результату.

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

---

## 2026-05-23 - StructXAttn Residual Fusion Stabilization

- `topic`: architecture stabilization
- `impact`: critical before interpreting StructXAttn drift results
- `status`: implemented as configurable merge modes
- `related_runtime`: `model.struct_xattn_merge_mode`

**Note (ukr):**

За результатами run `d9d019d3827548cc920554edbf47a75f` виявлено, що
поточний StructXAttn мав не лише проблему сили структурного сигналу, а
конкретний residual-normalization artifact:

```text
pre_norm_to_observed_ratio  ~= 0.25
post_norm_to_observed_ratio ~= 6.75
```

Це означає, що структурний delta до residual merge був помірним, але
`LayerNorm(node_hidden + delta)` роздував фактичну зміну hidden-state. Для
дисертації важливо не трактувати такі запуски як доказ того, що cross-attention
структура "погана"; спочатку треба відокремити дефект механіки злиття від
дефекту методології навчання.

Додано стабілізаційні режими:

1. `post_norm_residual` - історична поведінка для відтворення старих запусків;
2. `pre_norm_context` - рекомендований режим: нормалізується structural context
   до residual merge, а після merge не застосовується post-LN;
3. `residual_only` - ablation для перевірки ролі context normalization.

Також додано `model.struct_xattn_delta_ratio_max`, який може обмежити
`mean_abs(delta) / mean_abs(observed)`.

**Do not forget:**

- у Methodology/Limitations зазначити, що перші StructXAttn результати до
  стабілізації fusion не є фінальним висновком про непридатність cross-attention;
- окремо порівнювати `post_norm_residual` та `pre_norm_context`;
- для аналізу дивитися не тільки strict-F1, а й:
  `struct_xattn_pre_norm_to_observed_ratio`,
  `struct_xattn_post_norm_to_observed_ratio`,
  `drift_window_oos_confidence_mean`, `ECE`, `NLL`, `OOS`.

---

## 2026-05-24 - Dynamic Candidate Set And Topology Drift Audit

- `topic`: methodology limitation / audit evidence
- `impact`: critical for business-valid zero-shot claims
- `status`: audit tool implemented
- `related_runtime`: `tools/audit_topology_drift.py`

**Note (ukr):**

У реальних бізнес-версіях процесу нова topology може не лише змінювати
ймовірності переходів, а й додавати/прибирати BPMN-вузли та нові activity
labels. Це означає, що поточна fixed parametric classifier head
`Linear(hidden_dim, num_classes)` може бути методологічно недостатньою для
zero-shot сценарію: структура нової версії вже знає новий кандидат, але
класифікатор або не має для нього навченого виходу, або має майже нульовий
supervised signal.

Додано offline audit, який порівнює gateway-collapsed BPMN topology між
версіями та атрибутує prediction errors:

```powershell
.\.venv-modern\Scripts\python.exe tools\audit_topology_drift.py `
  --mlruns-dir mlruns `
  --experiment-id 854778689611649472 `
  --baseline-run <baseline_eval_drift_run_id> `
  --structural-run <structural_eval_drift_run_id> `
  --train-run <train_run_id> `
  --output-dir outputs\audits\topology_drift
```

Перший smoke audit для `5af1d05da0524106b4031dd7c39767c6` показав, що у
`loan_v4_re_pl_cb.bpmn` з'являється новий BPMN activity id
`Activity_07k7tyn` (`Monitor Ten`) і нові gateway-collapsed transitions:

```text
t_confirm_payout -> Activity_07k7tyn
Activity_07k7tyn -> t_update_risk
```

**Do not forget:**

- у Methodology описати, що topology drift змінює не лише feature distribution,
  а й candidate set;
- окремо розрізняти:
  1. known-label route change,
  2. added/removed BPMN node with known label,
  3. added/removed activity class;
- якщо errors концентруються на added/new candidates, наступним архітектурним
  кроком має бути dynamic topology-conditioned candidate scorer або
  semantic-topological retrieval head, а не лише stronger fusion scale;
**2026-05-24 update:** `prefix_last_activity` додано в future trace payload
через `prefix_last_activity_idx`; для старих run artifacts audit може
відновлювати його з XES за `(trace_idx, prefix_len)`. Повторний audit
`loan_xattn_5af1d05_prefix_last_activity` показав, що для sampled traces:

- `changed_transition_zone`: 2;
- `strict_error_but_allowed`: 20;
- `oos_prediction`: 13;
- `correct`: 5.

Це підтверджує, що частина post-drift помилок уже прив'язується до конкретної
зони зміни маршруту, але домінують OOS та allowed-but-wrong помилки. Для
дисертації це важливо трактувати як candidate-set/topology alignment problem,
а не лише як проблему сили fusion signal.

---

## 2026-05-24 - EOPKGTopologyConditioned Stage 1

- `topic`: topology-conditioned candidate scoring
- `impact`: important architecture step beyond fixed-head fusion modes
- `status`: Stage 1 implemented
- `related_runtime`: `model.type=EOPKGTopologyConditioned`

**Note (ukr):**

Додано окремий тип моделі `EOPKGTopologyConditioned`. Це не ще один
`fusion_mode`, а перехід до іншого принципу: структура задає простір
кандидатів, а observed prefix порівнюється з candidate embeddings. Stage 1
зберігає сумісність з поточним trainer/evaluator contract і повертає
`[B, C_train]`, але всередині уже рахує `node_scores [B, |V|]` і агрегує
дублікати BPMN-вузлів у activity class через MIL.

**Do not forget:**

- у тексті дисертації розділити fixed-head fusion ablations та
  topology-conditioned candidate scoring;
- Stage 1 ще не доводить повний zero-shot для нових activity classes, бо
  фінальний contract залишається `[B, C_train]`;
- ключові діагностики для аналізу:
  `candidate_node_score_mean_abs`, `candidate_class_score_mean_abs`,
  `duplicate_candidate_count_max`, `candidate_temperature`,
  `candidate_prediction_entropy`, `candidate_score_gap`;
- наступний методологічний крок: dynamic candidate contract `[B, C_v]` і
  semantic-topological candidate prototypes.

---

## 2026-05-24 - Research Path From Baseline To Topology-Conditioned Scoring

- `topic`: dissertation reasoning / model evolution
- `impact`: helps explain why simple structural fusion is not enough for zero-shot adaptation
- `status`: notes for future writing
- `related_runtime`:
  - `BaselineGATv2`
  - `EOPKGGATv2`
  - `EOPKGTopologyConditioned`

**Short thesis (ukr):**

Шлях дослідження йде не від "baseline гірший, структура краща", а від
послідовної перевірки, на якому саме рівні структура має входити в модель,
щоб бути корисною при drift:

```text
Baseline observed-only
  -> EOPKG fixed-head fusion modes
  -> topology-conditioned candidate scoring
  -> future dynamic candidate contract / semantic-topological retrieval
```

**Baseline:**

Baseline використовує тільки observed prefix / IG:

```text
P(y | observed_prefix)
```

Він не знає нову BPMN topology явно, але іноді краще переноситься між
версіями, бо не прив'язаний до застарілих структурних кореляцій. Це важливий
контрольний результат: слабший inductive bias може бути стабільнішим, якщо
структурний bias неправильно інтегрований.

**EOPKG fixed-head fusion modes:**

Перевірені режими `ClassMeanAttention`, `ClassMeanConcat`,
`StructuralPriorEncoder`, `TopologyStateEncoder`, `TopologyStateGraphEncoder`,
`ClassAwareStructuralScoring`, `StructXAttn`.

Їх спільне обмеження: фінальна prediction head здебільшого залишається
фіксованою:

```text
logits = f(observed_prefix, topology_signal), logits in R^{C_train}
```

Тобто структура додається як context/prior/logit correction, але не змінює
сам простір кандидатів. Якщо у новій версії процесу з'являється новий вузол,
зникає старий або змінюється маршрут, fixed head все одно прогнозує в межах
старого `C_train`.

**Main observation:**

Проста подача структури може:

- зменшувати OOS або покращувати mask/calibration diagnostics;
- іноді стабілізувати non-strict metrics;
- давати локальні покращення на окремих версіях;
- але не гарантує zero-shot improvement для strict-F1 після topology drift.

Причина: модель вчиться не "користуватися новою topology як умовою", а часто
лише підлаштовує fusion під topology, яку бачила під час train.

**Why simple structure does not solve zero-shot:**

1. `Conditional Structure Shift`:

```text
P_train(Y | X, A_v1) != P_test(Y | X, A_vN)
```

Однаковий або схожий prefix може мати інший valid next candidate у новій
версії. Якщо модель вивчила стару кореляцію, структура може стати джерелом
negative transfer.

2. `Fixed-head limitation`:

```text
classifier: R^D -> R^{C_train}
```

Нова topology може містити candidate, якого немає у виході classifier. Тоді
модель фізично не може чесно вибрати цей candidate без зміни prediction
contract.

3. `Fusion-level ambiguity`:

Структура, додана через attention/concat/logit scoring, може бути:

- занадто слабкою і ігноруватися;
- занадто сильною і псувати calibration/NLL/OOS;
- корисною лише як mask/filter, а не як learned candidate reasoning.

4. `Mask-guided effect is separate`:

`mask_guided_enabled=true` може покращити метрики, але це не доводить, що
структурний encoder навчився прогнозувати. Це доводить, що topology constraint
корисний як downstream filter. Тому треба порівнювати:

```text
TopologyConditioned-UN = candidate scoring only
TopologyConditioned-MN = candidate scoring + mask-guided constraint
```

Якщо приріст є тільки в `MN`, то основний ефект дає маска, а не модельне
розуміння структури.

**Why EOPKGTopologyConditioned is the next step:**

Нова модель змінює питання з:

```text
Which class from C_train should fixed classifier predict?
```

на:

```text
Which candidate from current topology best matches observed_prefix?
```

Stage 1:

```text
z_prefix = ObservedEncoder(IG_prefix)
h_j = TopologyCandidateEncoder(node_j, topology_edges, optional struct_x)
score_j = sim(z_prefix, h_j)
class_score_c = MIL_pool({score_j | node_to_class(j)=c})
```

Тут `struct_x` є enrichment для candidate node, а не головний IG-вхід.
Головний IG-вхід:

```text
x_cat, x_num, edge_index, batch
```

Мінімальний structural input:

```text
structural_edge_index + struct_node_to_class_index
```

Optional enrichment:

```text
struct_x = stats/topology node features
```

**Current limitation of Stage 1:**

Stage 1 все ще повертає:

```text
[B, C_train]
```

Тому він є bridge між старим pipeline і майбутнім dynamic candidate prediction.
Він потрібен, щоб перевірити сам принцип candidate scoring без повної
перебудови evaluator/checkpoints.

**Required future step:**

Для бізнес-валідної zero-shot адаптації потрібен Stage 2:

```text
candidate_logits: [B, C_v]
candidate_ids: candidates(topology_v)
prediction = argmax_{j in candidates(topology_v)} score_j
```

І далі semantic-topological prototypes, щоб новий candidate мав embedding до
появи логів:

```text
h_j = GNN_struct(Text/BPMN_metadata_embedding_j, topology_v)
```

**Dissertation framing:**

Не формулювати проміжний висновок як "структура не працює". Коректніше:

```text
Simple structural fusion is insufficient for zero-shot adaptation under
candidate-set drift and conditional structural shift.
```

Потрібна методологія, де topology:

1. задає candidate space;
2. входить у prediction як умова;
3. підтримує added/removed candidates;
4. окремо контролюється через calibration/OOS/mask diagnostics;
5. відділяється від downstream mask filtering.

---

## 2026-05-25 - TopologyConditioned Drift Improves Reliability More Than Strict-F1

- `topic`: EOPKGTopologyConditioned Stage 1 drift interpretation
- `run_id`: `2c42cdcae0264f93968330c012caf073`
- `baseline_reference`: `_Base-UN-drift-42` comparable versioned split runs
- `impact`: important for business-facing evidence and methodology framing
- `status`: preliminary observation, validate with seeds

**Observation (ukr):**

`EOPKGTopologyConditioned` Stage 1 на drift `v1..v4` не дає суттєвого приросту
`strict_macro_f1` відносно baseline, але помітно покращує reliability-метрики:

```text
EOPKGTC drift last:
strict_macro_f1 ~= 0.571
macro_f1        ~= 0.780
OOS             ~= 0.120
ECE             ~= 0.063
NLL             ~= 0.480

Comparable Base-UN drift last:
strict_macro_f1 ~= 0.578
macro_f1        ~= 0.767
OOS             ~= 0.134
ECE             ~= 0.154
NLL             ~= 1.079
```

**Interpretation (ukr):**

Topology-conditioned candidate scoring поки не вирішує exact next-activity
ranking після drift, але робить прогноз менш токсичним для бізнесу:

- менше виходів за дозволений простір (`OOS`);
- краща калібровка (`ECE`);
- нижчий negative log-likelihood (`NLL`);
- non-strict `macro_f1` трохи кращий, ніж baseline.

Це підтримує тезу, що topology уже корисна як reliability / governance signal,
але не достатня як повна zero-shot ranking methodology.

**Methodology consequence:**

У дисертації розділяти:

```text
1. business validity / reliability:
   OOS, ECE, NLL, pred_in_mask_rate

2. exact predictive accuracy:
   strict_macro_f1, strict_error_but_allowed_rate
```

Stage 1 може бути доказом користі структури для першого блоку, але для другого
потрібні Stage 2 dynamic candidates, richer candidate features/statistics або
окрема методологія ranking adaptation.

---

## 2026-05-25 - Dissertation Notes Capture Rule

- `topic`: research process hygiene
- `impact`: keep business/methodology conclusions from being lost between runs
- `status`: active working rule

**Rule (ukr):**

Усі висновки з експериментів, які важливі для бізнесової інтерпретації,
методології навчання, дисертаційної аргументації або майбутньої Scopus статті,
треба заносити сюди короткими тезами. Не чекати фінального тексту розділу.

**Examples of conclusions to capture:**

- що саме покращує topology: OOS, calibration, allowed-space, strict-F1,
  drift slope, confidence, або лише mask-filtering;
- де структура шкодить: overconfidence, NLL, negative transfer, wrong candidate
  ranking;
- які обмеження важливі для бізнесу: new/removed candidates, ambiguity,
  duplicate activities, cold-start topology before logs;
- які висновки впливають на методологію навчання або архітектуру.

---

## 2026-05-25 - Topology Helps Allowed-Space But Not Ambiguous Exact Ranking

- `topic`: EOPKGTopologyConditioned Stage 1 train run interpretation
- `run_id`: `0d2fd3875d7444f4a4be69ff3fb165e2`
- `impact`: important for business and methodology framing
- `status`: preliminary observation, validate on drift and seeds

**Observation (ukr):**

Для `EOPKGTopologyConditioned` на train `v1,v2` без mask-guided inference і без
statistics topology допомагає тримати прогноз у дозволеній області:

```text
test_oos = 0
target_in_mask_rate = 1
pred_in_mask_rate = 1
```

Але це не гарантує exact next activity у неоднозначних префіксах:

```text
strict_test_macro_f1 ~= 0.836
test_macro_f1 = 1.0
test_ambiguous_prefix_rate ~= 0.473
test_strict_error_but_allowed_rate ~= 0.056
```

**Interpretation (ukr):**

Topology добре задає allowed candidate space, але якщо prefix допускає кілька
валідних продовжень, сама topology не завжди містить достатньо інформації для
точного ранжування конкретного next event. Для бізнесу це означає: структура
корисна як обмеження простору рішень і audit/control layer, але для exact
prediction потрібна додаткова методологія ранжування кандидатів: статистика,
контекст префікса, temporal/process state, або candidate-level навчання.

**Methodology consequence:**

Розділяти дві задачі:

```text
1. valid candidate containment:
   y_hat in allowed(topology, prefix)

2. exact candidate ranking:
   y_hat == y_true inside allowed(topology, prefix)
```

Перша задача може покращувати OOS/calibration/business validity. Друга задача
потребує сильнішого навчання та/або Stage 2 dynamic candidate contract.
---

## 2026-05-25 - Dynamic Candidate Contract Foundation

- `topic`: EOPKGTopologyConditioned Stage 2 foundation
- `runtime_change`: added model-level `forward_candidate(contract)`
- `contract`: `candidate_logits [B, C_v]`, `candidate_class_index [C_v]`,
  `node_logits [B, |V|]`, `node_to_candidate_index [|V|]`

Ключова методологічна зміна: модель тепер може представити prediction space як
множину кандидатів поточної topology version, а не тільки як fixed activity vocab
`C_train`. Це ще не повний zero-shot для нових BPMN вузлів, бо стандартний
trainer/evaluator поки використовує backward-compatible `forward() -> [B, C_train]`.
Але це перший технічний крок до доведення гіпотези:

```text
new topology defines candidates first;
logs arrive later;
model scores candidates from topology rather than only selecting from old fixed head.
```

Для тексту дисертації важливо розділяти:

- `Stage 1`: topology node scoring + MIL pooling back to fixed labels;
- `Stage 2 foundation`: model can return dynamic topology candidate logits `[B, C_v]`;
- `Stage 2 end-to-end`: future trainer/evaluator must train and evaluate directly
  on dynamic candidate outputs.

**2026-05-25 update:** `training.dynamic_candidate_contract_enabled=true`
connects this contract to train/eval/drift. The trainer calls
`forward_candidate()` and optimizes topology-local `candidate_logits [B, C_v]`,
then projects them back to sparse fixed-label logits `[B, C_train]` so existing
CE, mask, calibration, and drift metrics still work. This is an important
transitional step: the loss now follows topology-defined candidates, but the
evaluation contract still cannot fully represent a new class outside
`C_train`. In dissertation terms, this should be described as
`Stage 2 compatibility`, not yet full semantic-topological zero-shot retrieval.

**2026-05-25 run observation:** for `_EOPKGTC-UN`
(`train_run=2b1bc6597752412a915a08e1778d1ed2`) and `_EOPKGTC-UN-drift`
(`drift_run=8061e9c61a7a43f7b6434d1c68002e37`), Stage 2 compatibility produced
graphs identical to Stage 1. Diagnostics explain why:

```text
vocab_activity_size = 37
candidate_dynamic_count = 37
duplicate_candidate_count_max = 1
candidate_class_score_mean_abs == candidate_node_score_mean_abs
```

In this configuration, the topology-local candidate set is effectively the same
as the fixed training vocab, and there are no duplicate candidate classes in the
pooled view. Therefore:

```text
forward() class pooling
==
forward_candidate() dynamic pooling + projection back to C_train
```

Methodology consequence: this step validates wiring but does not yet test the
business-critical zero-shot claim. To observe a difference, the evaluator/loss
must stop projecting back to `C_train` and operate on true candidate ids /
BPMN node ids, especially where a topology version adds/removes candidates.

**2026-05-25 Stage 2 true plan correction:** true candidate-id learning cannot
reuse the old "first structural payload per batch" assumption. Once prediction
space becomes topology-local `[B, C_v]`, mixed-version/mixed-topology batches
can silently corrupt the candidate axis. The methodology must either enforce
homogeneous topology batches (`single_topology_required`) or explicitly split
mixed batches by topology before loss/evaluation.

Additional interpretation guardrails:

- MIL/set-aware loss over duplicate BPMN candidates must log target-set entropy
  and variance. If duplicate target logits remain identical, the model has not
  learned to distinguish duplicate structural nodes.
- Missing target candidates are not normal noise. A high
  `candidate_missing_target_rate` means log/topology alignment is broken, so
  training must fail instead of continuing with zero-gradient rows.

**2026-05-25 batching correction:** для true candidate-id режиму недостатньо
лише падати на mixed batch. Експеримент має бути прохідним, тому DataLoader
повинен формувати topology-homogeneous mini-batches за:

```text
process_version_idx + stats_snapshot_version_idx
```

Якщо для певної topology identity є лише 1 prefix/sample, batch size для цієї
групи буде `1`. Це не недолік, а умова коректності: candidate axis `[C_v]`
залежить від версії процесу і stats/as-of snapshot. Змішування різних axes в
одному forward/loss робить висновки про zero-shot користь структури
методологічно невалідними.

**2026-05-25 Stage 2 true candidate-level path:** `candidate_contract_mode=candidate_id`
now trains on topology-local `candidate_logits [B, C_v]` with set-aware CE:

```text
target_candidate_mask[b, j] = candidate_class_index[j] == y_b
loss_b = -logsumexp(logits_b[target_candidate_mask_b]) + logsumexp(logits_b[:])
```

This is methodologically important: when the log target is an activity label
but the topology contains multiple BPMN nodes for the same activity, the model
is not forced to choose one arbitrary duplicate node. All candidates with the
target class are accepted. This directly addresses duplicate activity identity
ambiguity at the class-label level, while leaving exact BPMN node target
learning for future log sources that can provide node ids.

New audit metrics to mention in dissertation experiments:

- `candidate_target_in_candidate_set_rate`: whether the topology candidate set
  contains the target activity label;
- `candidate_missing_target_rate`: alignment failure rate between logs and
  topology candidate space;
- `candidate_target_duplicate_count_max`: duplicate candidate pressure;
- `candidate_target_set_logit_variance_mean` and
  `candidate_target_set_entropy_mean`: whether the model learns to distinguish
  duplicate structural candidates or keeps them collapsed.

If `candidate_missing_target_rate` is high, the run should be treated as invalid
alignment rather than weak model quality.

**2026-05-26 EOPKGTopologyConditioned run analysis:** candidate-level wiring is
valid, but the tested XES/collapse configuration still does not demonstrate
topology-based zero-shot transfer. In runs
`0f36f16dcd484c03b454c201d2723d14` and
`709bd0a6f26f4940b014c8d14b929584`, diagnostics show:

```text
candidate_dynamic_count = 37
duplicate_candidate_count_max = 1
candidate_target_in_candidate_set_rate = 1.0
candidate_missing_target_rate = 0.0
```

This means the topology-local candidate axis effectively equals the old
activity-label vocabulary. The loss path changed, but the model still answers
mostly "which activity label?", not "which BPMN candidate node in this version?".

Important methodology conclusion: for the dissertation claim about using a new
topology before new logs arrive, the next architecture step must expose stable
BPMN candidate ids and candidate-level diagnostics. Activity-label candidate
scoring is useful as a compatibility bridge, but it is insufficient evidence for
business-valid zero-shot adaptation under added/removed BPMN nodes.

Trace evidence from drift run `709bd0a6f26f4940b014c8d14b929584`:

```text
v1/v2 sampled errors: mostly strict_error_but_allowed
v3/v4 sampled errors: OOS appears strongly
```

Interpretation: topology/class candidates can keep predictions in a broadly
valid area, but future-version route changes still cause exact-target failure
and OOS when the candidate identity contract is collapsed to labels.

**2026-05-26 topology-flow loss update:** after confirming that the tested loan
versions have `duplicate_candidate_count_max=1` and a stable candidate count,
the main failure hypothesis shifted from MIL/duplicate ambiguity to ignored
flow changes. A new candidate-level topology-flow penalty was added for
`EOPKGTopologyConditioned + candidate_id`.

Key dissertation point:

```text
Topology as candidate set is necessary but not sufficient.
The learning objective must also penalize probability mass on candidates
forbidden by the current topology flow.
```

Metrics to report for this step:

- `candidate_oos_rate`;
- `candidate_invalid_probability_mass`;
- `candidate_valid_probability_mass`;
- `candidate_valid_invalid_logit_margin`;
- `topology_conditioned_topology_flow_penalty_loss`.

Interpretation for business framing: topology can be used not only as an
additional feature, but as a validity surface over the candidate space. If the
model remains close to baseline after this penalty, the next methodological
step should focus on edge-aware candidate scoring / flow-conditioned routing,
not on stronger late fusion.

**2026-05-26 topology-native candidate identity correction:** the latest audit
showed that the current `candidate_id` path is still partially tied to the
training activity vocabulary. With `version_scope_policy=train_cut`, future
activities outside the training versions can be collapsed to `<UNK>`, and
topology edges containing those activities can be skipped during topology
compilation. This means the model is not yet forced to choose from the real
candidate set of the new BPMN version.

Methodology consequence:

```text
Fixed activity-vocab candidate scoring is not sufficient evidence for
business-valid zero-shot topology transfer.
```

For the dissertation, the next architecture step should be described as a move
from fixed-label classification to topology-native candidate scoring:

```text
prefix state z_t + current topology candidates K_v
    -> score every BPMN candidate of version v
    -> project to fixed labels only for compatibility metrics
```

This directly supports the main hypothesis: when a new process version appears,
the model must be able to use the new topology before logs of that version are
available. If future candidates are absent from the candidate universe, the
hypothesis cannot be evaluated correctly.

**2026-05-26 implementation note:** topology-native candidate identity was
implemented as the next experimental contract. The model can now expose
topology-local candidates whose labels are absent from the train activity
vocabulary:

```text
candidate_class_index = -1
```

This changes the interpretation of experiments:

- fixed-label F1 remains a compatibility metric;
- candidate-space metrics become primary for zero-shot claims;
- unseen candidates must be visible in traces/MLflow before discussing whether
  the model transfers topology knowledge;
- if unseen candidates are visible but never selected, the next blocker is
  semantic grounding / retrieval-style candidate prototypes, not basic topology
  wiring.

**2026-05-31 impulse activation routing implementation note:** the
`EOPKGTopologyConditioned` candidate path now has an experimental dynamic
topology-routing mode:

```yaml
model.topology_conditioning_mode: impulse_activation_routing
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
```

This mode keeps the observed prefix encoder and injects the current execution
pointer from `struct_prefix_state_x` into topology candidate nodes before
structural message passing. The model then scores topology-local candidates
directly in `[B, K_v]` space. It is therefore a test of whether dynamic
topology routing changes candidate ranking, not another fixed-vocabulary
fusion mode.

Dissertation interpretation: this is a planned hypothesis test for whether the
current topology can route candidate ranking after structural drift. Success
should be judged by candidate-space diagnostics, OOS/calibration, and drift
degradation slope, not by a single fixed-label F1 spike.

**2026-06-01 follow-up note: cyclic route disambiguation:** current
`impulse_activation_routing` is primarily node-state based. It can highlight
the last executed node, visit count, and recency, but cyclic processes may
require path/edge state to distinguish prefixes that touch the same nodes
through different loop routes.

Next-step hypothesis after stabilizing impulse routing:

```text
node-level impulse tells the model where the process is;
edge/path-aware prefix state tells the model how it arrived there.
```

Candidate future features:

- `prefix_edge_traversal_count`;
- `prefix_last_edge_or_transition`;
- `prefix_visit_count_per_node`;
- `prefix_last_visit_distance`;
- `loop_iteration_estimate`.

This should be positioned as a later extension for cyclic BPMN/EOPKG behavior,
not as part of the current stabilization run.

**2026-06-01 follow-up note: impulse propagation depth:** `impulse_gnn_layers`
should be described as a propagation-depth parameter for the execution impulse,
not only as a generic GNN hyperparameter.

Interpretation:

```text
impulse_gnn_layers = how many topology hops the current execution pointer can
influence before candidate scoring
```

Recommended framing:

- `XES + gateways: collapse_for_prediction`: business activity transitions are
  already projected to prediction nodes, so `impulse_gnn_layers=1` is the
  default candidate;
- `Camunda-like logs + gateways: preserve`: technical/service/gateway nodes can
  sit between two business activities, so `impulse_gnn_layers=2` becomes a
  meaningful ablation because the impulse may need to cross an intermediate
  technical node;
- `impulse_gnn_layers >= 3` should be treated cautiously because it can spread
  the execution pointer too broadly and create over-smoothing.

Metrics for this explanation:

- `impulse_gnn_oversmoothing_ratio`;
- `candidate_prediction_entropy`;
- `candidate_score_gap`;
- `strict_error_but_allowed_rate`;
- `test_nll` / `test_ece`.

**2026-06-02 parallel process-state mask correction:** complex parallel XES
experiments showed that a last-completed-node mask is not sufficient for
parallel/interleaved execution. The issue is not wrong version grouping:
event-level and trace-level `concept:version` are consistent. The issue is that
the next completed task may already be active in another parallel branch and is
therefore not a direct successor of the last completed task.

Important dissertation point:

```text
For parallel BPMN execution, the structural candidate mask must be based on
process state, not only on direct successors of the last completed activity.
```

Corrected mask formulation:

```text
M_t = A_t union S_t
```

where:

- `A_t` is the active not-yet-completed task set reconstructed from lifecycle
  events or provided by BPMS/simulator metadata;
- `S_t` is the newly enabled structural successor set.

Impact on interpretation:

- low `target_in_mask_rate` on complex parallel data can indicate incomplete
  state abstraction rather than invalid BPMN topology;
- `experiment.mask_guided_enabled` cannot fix missing targets if the mask
  itself omits active parallel tasks;
- `_EOPKG_TC` should inject impulse not only into `last_completed`, but also
  into active candidate nodes;
- strict next-activity F1 should be reported together with
  `target_in_mask_rate`, `pred_in_mask_rate`, `strict_error_but_allowed_rate`,
  and a parallel ambiguity/active-set diagnostic.

Implementation plan:

```text
docs/worklogs/MVP2_5_Parallel_Process_State_Mask_Plan_2026-06-02.MD
```

**2026-06-03 state-aware relaxed reachability note:** lowering relaxed mask
aggressiveness by parameters alone is not sufficient. The core issue is not only
mask size, but stale candidate semantics: relaxed reachability can keep
parallel sibling tasks that were valid from an older split anchor but are
already completed in the current prefix.

Dissertation interpretation:

```text
A topology-derived mask can reduce OOS while still hurting strict next-activity
prediction if it over-expands the allowed set with stale completed candidates.
```

Therefore, process-state masks must report both inclusion and over-inclusion:

- `target_in_mask_rate` shows whether the true target is reachable by the mask;
- `test_oos` shows how often the model leaves the topology-derived allowed set;
- `strict_error_but_allowed_rate` shows whether the mask is too broad for exact
  next-activity prediction;
- `process_state_mask_target_suppressed_by_completed_filter_rate` is a safety
  check: it must stay `0`, otherwise completed-candidate suppression is too
  aggressive;
- `process_state_mask_completed_suppression_rate` shows whether the refined
  relaxed mask actually removes stale candidates.

This supports the broader methodology point: for parallel/interleaved BPMN logs,
structural guidance needs process-state semantics, not only graph reachability.
