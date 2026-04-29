# Architecture Debt

Active architecture debt index for `bpm_prediction`.

This file is current-state documentation. It replaces direct agent dependency on
historical debt worklogs.

---

## Metadata

- `status`: active
- `audience`: human-and-agent
- `source_of_truth`: true
- `language_policy`: keys and section headers in English, human descriptions in Ukrainian
- `last_updated`: 2026-04-29

---

## Debt Status Values

- `active`: debt exists and affects future design or experiments.
- `proposed`: target direction exists in ADR, but implementation is not accepted.
- `deferred`: acknowledged, but not scheduled for immediate work.
- `blocked`: cannot be resolved until another decision or implementation exists.
- `closed`: debt is resolved; keep only if historical trace is useful.

---

## P0 Research-Grade Debt

### snapshot_homogeneous_batching

- `status`: active
- `priority`: P0
- `adr`: `docs/adr/0005-snapshot-homogeneous-batching.md`
- `current_behavior`: mixed snapshot batches may warn and use first graph structural payload
- `target_state`: one structural forward context uses one stats snapshot identity

**Description (ukr):**

Зараз PyG batch може містити графи з різних stats snapshots. Runtime попереджає
про це, але для structural tensors бере payload першого графа в batch. Це
прийнятний тимчасовий workaround, але не чистий контракт для строгих temporal або
drift-досліджень: один forward може частково представляти кілька різних
`knowledge_version` / `as_of` станів.

**Impact (ukr):**

Фінальні висновки про вплив структури або дрейфу можуть бути важче захищати,
бо structural branch в одному batch не гарантує єдиного часово-структурного
контексту.

**Next direction:**

Implement snapshot-homogeneous batching through sampler bucketing,
micro-batching, or fail-fast research profile.

---

### activity_to_topology_alignment_gate

- `status`: closed
- `priority`: P0
- `adr`: `docs/adr/0006-research-grade-activity-topology-alignment-gate.md`
- `current_behavior`: producer-side alignment profiles and strict research profile are implemented
- `target_state`: closed; maintain catalog/docs sync for future alignment config keys

**Description (ukr):**

Структурна гілка корисна тільки тоді, коли `activity_id` з логів коректно
співпадають з BPMN/topology node ids і stats indexes. Якщо alignment слабкий,
`allowed_target_mask` і `struct_x` можуть формально існувати, але нести
спотворений або майже нульовий структурний сигнал.

**Impact (ukr):**

Є ризик тихої деградації: експеримент виглядає як EOPKG/structural run, але
модель фактично отримує некоректно вирівняний структурний контекст.

**Next direction:**

Use `sync_stats.alignment_gate.profile: research_strict` with `on_fail: raise`
for dissertation-grade runs. Future work belongs to
`topology_projection_alignment`, not this producer-side gate.

---

### topology_projection_alignment

- `status`: active
- `priority`: P0
- `adr`: `docs/adr/0007-topology-projection-alignment.md`
- `current_behavior`: projection diagnostics, cache node-metadata fingerprint, strict `on_fail`, forward-stat counters, and EOPKGGATv2 structural index fail-fast are implemented and pytest-verified
- `target_state`: residual work only if a dedicated run-level JSON artifact beyond forward logs is required

**Description (ukr):**

Режим `collapse_for_prediction` змінює topology view для прогнозування, наприклад
може прибирати або згортати технічні вузли. Після такої модифікації треба
довести, що індекси активностей, маски, structural edges, `struct_x` rows і
edge weights залишилися узгодженими. Інакше структура може бути правильною до
projection, але некоректною після неї.

**Impact (ukr):**

Можливі помилки, де mask або structural tensors посилаються на неправильний
індекс/вузол після згортання topology. Це критично для research-grade оцінки
structural awareness.

**Next direction:**

Use `mapping.graph_feature_mapping.topology_projection.on_fail: raise` for
research-grade runs. Keep residual work only if a dedicated run-level JSON
artifact beyond forward logs is required.

---

### duplicate_activity_identity_ambiguity

- `status`: active
- `priority`: P0
- `adr`: none
- `current_behavior`: activity classes are usually keyed by log activity label/name, while BPMN structure can contain multiple distinct nodes with the same label
- `target_state`: explicit identity model that separates `activity_label` from stable BPMN/log node identity, or strict diagnostics that mark duplicate labels as ambiguous

**Description (ukr):**

Якщо в BPMN є дві різні задачі з однаковою назвою, а в event log є тільки
`concept:name` / activity label, pipeline бачить їх як один клас у
`activity_vocab`. BPMN-структура при цьому розрізняє node ids, але target `y`,
stats і activity-level mask не мають стабільної інформації, яка саме BPMN-нода
була виконана.

Це не є частиною `parallel target-mask semantics`. Паралельність може зробити
симптоми помітнішими, але root cause інший: немає однозначної identity mapping
між log event і BPMN prediction node.

**Impact (ukr):**

Статистика для однакових назв змішується, `allowed_target_mask` може дозволити
тільки "активність з такою назвою", а не конкретну структурну ноду, і `struct_x`
може агрегувати сигнали з кількох різних BPMN-позицій. У research-grade
експериментах це робить структурний сигнал неоднозначним: модель може
передбачати правильну назву, але не правильну структурну позицію.

**Relationship to other debt:**

- `topology_projection_alignment`: must stay compatible with future
  node-identity-aware mapping, but should not solve duplicate labels now.
- `target_mask_parallel_semantics`: should be handled only after activity
  identity ambiguity is either resolved or explicitly accepted as label-level
  prediction.

**Next direction:**

Add diagnostics that detect non-injective mapping from BPMN prediction nodes to
activity labels. Later choose target identity policy:

1. label-level prediction only, where duplicate BPMN nodes are accepted but
   structural interpretation is limited;
2. node-level prediction with `bpmn_node_id` / stable activity id in logs;
3. route-inferred node identity as an explicit heuristic with confidence and
   audit output.

---

## P1 Maintainability Debt

### cli_composition_root_overgrowth

- `status`: active
- `priority`: P1
- `adr`: `docs/adr/0008-cli-composition-root-boundary.md`
- `current_behavior`: `src/cli.py` combines composition, config, data prep, cache, telemetry, and wiring
- `target_state`: `src/cli.py` remains thin composition root

**Description (ukr):**

`src/cli.py` історично виріс і зараз містить не тільки wiring, а й значну частину
runtime orchestration: config overrides, trace preparation, split/cascade,
graph dataset cache, telemetry/profile, model/trainer setup. Це не обов'язково
порушує dependency rules, але робить файл складним для аудиту та змін.

**Impact (ukr):**

Зміни в runtime поведінці важче локалізувати й тестувати. Є ризик, що нові run
modes або cache/reporting логіка ще більше збільшать coupling у CLI.

**Next direction:**

Plan a dedicated refactor: move data preparation, cache orchestration, and run
profile/report assembly into focused application-level services/helpers.

---

## Closed Documentation/Tooling Debt

### config_catalog_alignment_gate_gap

- `status`: closed
- `priority`: P1
- `adr`: none
- `previous_behavior`: `sync_stats.alignment_gate.*` was implemented in `tools/sync_stats.py` but missing from `configs/ui/config_catalog.yaml`
- `resolved_by`: added catalog entries for `sync_stats.alignment_gate.*` on 2026-04-27
- `target_state`: every implemented config key remains represented in `configs/ui/config_catalog.yaml`

**Description (ukr):**

У коді вже існує producer-side `sync_stats.alignment_gate.*`: `enabled`,
`warn_on_fail`, `min_event_match_ratio`, `min_unique_activity_coverage`,
`min_node_coverage`, `on_fail`. Під час audit було виявлено, що ці параметри не
були представлені в machine-readable config catalog. Це закрито додаванням
відповідних entries у `configs/ui/config_catalog.yaml`.

**Impact (ukr):**

Ризик для поточного стану знято. Залишковий ризик процесний: нові config keys
потрібно додавати в catalog у тому самому change.

**Next direction:**

Keep `configs/ui/config_catalog.yaml` synchronized with implemented config
keys. Add catalog entries in the same change whenever config keys are added.

---

## P2 Future MVP Debt

### mvp3_reliability_semaphore_not_ready

- `status`: deferred
- `priority`: P2
- `adr`: none
- `current_behavior`: MVP2.5 has quality diagnostics and OOS metrics, but no full Reliability Semaphore
- `target_state`: MVP3 defines OOD calibration, latent artifacts, and Green/Yellow/Red policy

**Description (ukr):**

MVP2.5 має частину бази для епістемічного контролю: quality diagnostics,
snapshot metadata, OOS-related metrics. Але повний Reliability Semaphore ще не
реалізований: немає калібрування OOD/Wasserstein порогів, pipeline для latent
representations і policy-мапи `Green/Yellow/Red -> runtime action`.

**Impact (ukr):**

Не можна заявляти готовність MVP3-рівня епістемічного контролю. Поточні
експерименти можуть підготувати дані для нього, але не замінюють сам Semaphore.

**Next direction:**

Create MVP3 ADR/spec after MVP2.5 research-grade blockers are closed.

---

## Historical Sources

Use these only as background context:

1. `docs/worklogs/MVP2_5_Canonical_Doc_Sync_and_Architecture_Debt_2026-04-24.MD`
2. `docs/worklogs/MVP2_5_Dissertation_Alignment_and_Blocking_Debt_Analysis_2026-03-21.MD`
3. `docs/worklogs/MVP2_5_Stage4_2_OptionA_Unbatch_Fix_Report.MD`
4. `docs/worklogs/mismatch_fix_plan.md`
5. `docs/worklogs/change_attention_aproach.md`

---

## Maintenance Rule

When debt changes:

1. update this file,
2. update the related ADR if target direction changed,
3. update `AGENTS.MD` only if routing or hard rules changed,
4. do not create a new worklog as source of truth.
