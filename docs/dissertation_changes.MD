# Dissertation Changes

Short checklist of project changes that should be reflected in the dissertation.

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
