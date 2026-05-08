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
