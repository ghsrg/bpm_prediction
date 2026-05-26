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
- `last_updated`: 2026-05-25

---

## Debt Status Values

- `active`: debt exists and affects future design or experiments.
- `proposed`: target direction exists in ADR, but implementation is not accepted.
- `deferred`: acknowledged, but not scheduled for immediate work.
- `blocked`: cannot be resolved until another decision or implementation exists.
- `closed`: debt is resolved; keep only if historical trace is useful.

---

## P0 Research-Grade Debt

### dynamic_candidate_prediction_contract

- `status`: active
- `priority`: P0
- `adr`: none
- `current_behavior`: `EOPKGTopologyConditioned` exposes model-level `forward_candidate(contract)` with per-topology candidate logits `[B, C_v]`, `candidate_class_index`, `node_logits`, and `node_to_candidate_index`; `fixed_projection` projects candidate logits back to sparse fixed-label logits `[B, C_train]`; `candidate_id` trains with set-aware CE directly on `[B, C_v]`, reports candidate target mapping diagnostics, and maps predictions/probabilities back to global activity labels for existing metrics/drift reporting
- `target_state`: train/eval/drift contracts consume per-version candidate logits `[B, C_v]`, stable candidate ids, BPMN node ids, and target candidate mapping for added/removed BPMN nodes without requiring fixed-vocab metric projection

**Description (ukr):**

Поточний prediction contract все ще орієнтований на фіксований activity vocab,
який існував під час навчання. Це обмежує бізнес-валідний zero-shot сценарій:
нова BPMN версія може додати або прибрати кандидатні вузли, але fixed-head
classifier не може чесно видати логіт для нового candidate без розширення
контракту. `EOPKGTopologyConditioned` Stage 2 compatibility path уже дозволяє
trainer/evaluator оптимізувати topology-local candidate scores, але результат
поки проектується назад у `[B, C_train]`.

**Impact (ukr):**

Без чистого candidate-id evaluation contract не можна робити сильну заяву, що
модель повністю підтримує нові активності до появи логів. Поточні експерименти
треба позначати як Stage 2 compatibility / fixed-label projected candidate
scoring, а не як повну semantic-topological zero-shot адаптацію.

**Next direction:**

Complete the remaining true candidate-id contract pieces:

1. keep current `fixed_projection` path for backward-compatible experiments;
2. add stable `candidate_ids` / BPMN node ids per topology version;
3. keep set-aware class-target mapping for duplicate activity labels, then add
   exact node-level targets when the log source can provide them;
4. compute candidate-level mask/OOS/calibration diagnostics directly on
   `[B, C_v]`;
5. keep backward-compatible fixed-label reporting for comparison with old runs.

---

### stats_backed_structural_payload_caching

- `status`: active
- `priority`: P0
- `adr`: none
- `current_behavior`: snapshot-aware stats payload cache, deduplicated shard payload format, and one-pass prebuilt drift-window evaluation are implemented; full real drift validation is still pending
- `target_state`: full `S_Str-USc-drift-loan` validation confirms no Neo4j defunct connection, no system-wide virtual-memory exhaustion, and no repeated graph rebuild during drift-window evaluation

**Description (ukr):**

У режимі `eval_drift` з `structural_mode=true` та `statistic_enabled=true`
багато prefix graphs можуть резолвитись в один і той самий stats snapshot, але
runtime все одно може повторно матеріалізувати важкі snapshot DTO payloads або
тримати їх у cache за занадто дрібним ключем exact `as_of_ts`.

Після цього кожен prefix graph також може нести власну копію structural payload:
`struct_x`, `structural_edge_index`, `structural_edge_weight`,
`struct_node_to_class_index` та snapshot metadata. Для одного
`process_version` і одного resolved `as_of_snapshot` цей payload часто
однаковий для великої кількості prefix graphs.

На повному drift-запуску це створює непропорційне навантаження на Neo4j,
Neo4j driver, RAM/virtual memory під час побудови test graphs і запису shards.
`graph_dataset_disk_spill` частково допомагає, але не усуває дублювання:
до flush/serialization у пам'яті все одно може накопичуватися багато об'єктів з
повторюваними DTO та structural tensors.

**Observed evidence (ukr):**

Проблема проявилась для `S_Str-USc-drift-loan` на повному
`loan_v1_v4_simulated` dataset: `Base-UN-drift`, `S_Str-UNc-drift` і
`S_Str-USc-loan` проходили, а комбінація full `eval_drift` + structure + stats
завершилась системним low virtual memory. Windows Event Log показав, що
`python.exe` спожив приблизно 32.8 GB virtual memory перед падінням.

**Impact (ukr):**

Stats-backed structural drift experiments можуть бути нестабільними або
непрохідними на повному dataset, навіть якщо логіка моделі й mapping правильні.
Це блокує надійне порівняння структурної статистики в dissertation-grade
drift-запусках без ручного зменшення `graph_dataset_shard_size`, `fraction` або
інших runtime-обмежень.

**Next direction:**

Implement this as one debt with two stages:

1. snapshot-aware DTO/stats payload cache: resolve heavy stats payloads by
   snapshot identity instead of exact prefix `as_of_ts`;
2. structural payload deduplication: introduce a shared structural payload
   registry/cache keyed by `(process_version, as_of_snapshot,
   topology_projection_fingerprint, stats_mapping_fingerprint)`.

Prefix graphs should store only a lightweight payload reference, or the
loader/collate path should attach the shared payload at batch time.

Keep this compatible with `snapshot_homogeneous_batching`: batching should not
mix incompatible payload identities unless the research profile explicitly
allows it.

Implementation plan:

`docs/worklogs/MVP2_5_Stats_Backed_Structural_Payload_Caching_Plan_2026-05-12.MD`

Implementation report:

`docs/worklogs/MVP2_5_Stats_Backed_Structural_Payload_Caching_Report_2026-05-12.MD`

Follow-up one-pass drift plan:

`docs/worklogs/MVP2_5_One_Pass_Drift_Window_Evaluation_Plan_2026-05-13.MD`

Follow-up one-pass drift report:

`docs/worklogs/MVP2_5_One_Pass_Drift_Window_Evaluation_Report_2026-05-13.MD`

---

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

### candidate_batch_topology_grouping

- `status`: active
- `priority`: P0
- `adr`: none
- `current_behavior`: minimal `candidate_contract_mode=candidate_id` behavior
  uses topology-homogeneous DataLoader batching by
  `process_version_idx + stats_snapshot_version_idx` for indexable graph
  sources, including in-memory datasets and `ShardedGraphDataset`; new sharded
  graph cache entries store lightweight `topology_segments` so sampler setup
  does not need to hydrate full graph payloads; manually mixed batches still
  fail via
  `training.candidate_batch_topology_policy: single_topology_required`
- `target_state`: trainer can split mixed batches by topology identity and run
  candidate forward/loss per group, then aggregate loss and metrics

**Description (ukr):**

У true candidate-id режимі різні версії процесу можуть мати різні candidate
axes `[C_v]`. Тому один forward/loss не може безпечно використовувати перший
structural payload для всього batch. Мінімальна реалізація має формувати
homogeneous batches за `process_version_idx + stats_snapshot_version_idx`, а
guard `single_topology_required` має зупиняти будь-який змішаний batch, який
все одно дійшов до trainer. Повна реалізація `group_by_topology` має вміти
приймати вже змішаний batch, розбивати його на sub-batches і агрегувати
loss/metrics.

**Impact (ukr):**

Без цього guard Stage 2 true candidate-id може давати тихо неправильний
loss/eval: targets і candidate logits будуть зіставлятися з topology axis не
тієї версії або snapshot. Це блокує валідну перевірку zero-shot гіпотези.

**Next direction:**

Implemented first-stage protection:

1. topology-homogeneous DataLoader batching for `candidate_id` mode;
2. `single_topology_required` fail-fast guard as a safety net;
3. config catalog entries for candidate contract and batch topology policy;
4. tests that legacy modes keep current behavior.

Remaining debt: implement `group_by_topology` as a separate controlled change
for cases where mixed batches are intentionally accepted and split inside the
trainer. Old sharded cache entries without `topology_segments` may still need a
one-time shard scan or cache rebuild before candidate-id eval starts.

Implementation plan:

`docs/worklogs/MVP2_5_Single_Topology_Required_Guard_Plan_2026-05-25.MD`

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

## P1 Research Methodology Debt

### version_order_metadata_policy

- `status`: deferred
- `priority`: P1
- `adr`: none
- `current_behavior`: `versioned_zero_shot` design assumes parseable ordinal process versions such as `v1`, `v2`, `v3`
- `target_state`: version order can be derived from explicit metadata such as `effective_from`, `deployment_ts`, or `topology_version_seq`

**Description (ukr):**

Перший дизайн `versioned_zero_shot` може рахувати `version_distance` через
порядковий номер версії: `v4 - v1 = 3`. Це достатньо для поточного
`loan_v1_v4_simulated`, але не є загальним рішенням для реальних BPM-систем,
де версії можуть мати назви, описи, release labels або timestamps без
простого ordinal id.

**Impact (ukr):**

Без явного metadata ordering майбутній `versioned_zero_shot` може неправильно
рахувати `version_distance`, future-tail order, known/future split або
degradation slope для процесів з непарсабельними назвами версій.

**Next direction:**

Add explicit version ordering metadata to topology/process-version DTOs and use
that metadata for `version_distance`, future-tail evaluation, and leakage
guards.

---

### topology_conditioned_extended_negative_sampler

- `status`: deferred
- `priority`: P1
- `adr`: none
- `current_behavior`: first `topology_conditioned` strategy is designed around `wrong_version_topology` plus `drop_edges_same_version`
- `target_state`: negative sampler can support mixed-batch subgroup contrast and additional corruption policies after first experiments demonstrate useful topology dependency

**Description (ukr):**

Перший варіант `topology_conditioned` має використовувати version-safe
negatives:

```text
wrong_version_topology only from known versions
drop_edges_same_version for local topology corruption
```

Більш агресивні corruption policies (`rewire_edges`, node feature corruption,
gateway policy corruption) свідомо не входять у перший scope, щоб не змішати
основну гіпотезу з важко інтерпретованими аугментаціями.

**Impact (ukr):**

Якщо `wrong_version_topology + drop_edges` покаже, що модель почала реагувати
на topology, наступним кроком може бути підсилення robustness через додаткові
corruption policies. Якщо додати їх одразу, буде складніше пояснити, який саме
механізм дав або не дав ефект.

**Next direction:**

Implement only `drop_edges` first. Add `rewire_edges` or richer corruption
policies later as a separate controlled experiment.

Also deferred:

- `mixed_batch_subgroup_contrast`: when one batch contains several process
  versions, compute wrong-version topology contrast per version subgroup instead
  of skipping the wrong-version term. This requires either per-version
  microbatch splitting or per-sample structural payload contracts.
- `soft_edge_drop`: keep `structural_edge_index` unchanged and corrupt
  `structural_edge_weight` instead of physically deleting edges. This requires
  the structural GNN forward path to consume edge weights; current
  `EOPKGGATv2` structural GNN does not.

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

### split_fraction_policy_duplication

- `status`: active
- `priority`: P1
- `adr`: none
- `current_behavior`: version-aware fraction/split policy is mirrored in `src/cli.py` and `src/application/use_cases/trainer.py`
- `target_state`: one application-level split/fraction policy service is used by CLI preparation and trainer preparation

**Description (ukr):**

Після додавання `experiment.fraction_strategy`, `experiment.split_strategy=versioned`
та `experiment.version_scope_policy` логіка cascade/split тимчасово існує у двох
місцях: CLI data preparation і `ModelTrainer`. Це зберігає сумісність поточного
пайплайна, але створює ризик розходження поведінки при майбутніх змінах.

**Impact (ukr):**

Якщо змінювати правила split/fraction лише в одному місці, train/eval або cache
fingerprint можуть працювати з іншою семантикою, ніж unit-тести trainer path.

**Next direction:**

Extract a focused application service for experiment trace selection:
`fraction_strategy`, `version_scope_policy`, macro cut, and micro split should
have one implementation and both CLI and trainer should delegate to it.

---

### desktop_ui_catalog_contract_drift

- `status`: active
- `priority`: P1
- `adr`: none
- `current_behavior`: legacy Tkinter desktop Experiment UI mixes catalog-driven dynamic forms with manually rendered Core fields; a parallel PySide6 registry-driven prototype exists but is not production run UI yet
- `target_state`: desktop UI renders Project Setup, Experiment Run, and Advanced fields from one field registry derived from `configs/ui/config_catalog.yaml`
- `audit`: `docs/worklogs/MVP2_5_Desktop_UI_Field_Dependency_Audit_2026-05-22.MD`
- `matrix`: `outputs/ui/desktop_ui_field_dependency_matrix.csv`

**Description (ukr):**

Desktop UI зараз має кілька source of truth для одного config key:
`configs/ui/config_catalog.yaml`, ручні Core widgets у `tools/experiment_ui.py`,
`self.vars`, logic завантаження YAML, defaults, generated config output та
enable/disable rules. Через це новий параметр може бути реалізований у runtime
і catalog, але не з'явитися на потрібній вкладці або не потрапити назад у YAML.

Потрібна явна ієрархія полів:

1. `Project Setup`: стабільні налаштування проекту, джерел логів, BPMN/Neo4j,
   adapter, mapping та stats source.
2. `Experiment Run`: параметри, які часто змінюються під час порівняння
   запусків: mode, checkpoint/retrain, fraction/split, structure, mask,
   statistics, `fusion_mode`, `learning_strategy`.
3. `Advanced`: рідкісні або mode-specific параметри для тонкого налаштування,
   debug/tracing, auxiliary losses, dataloader/performance knobs.

**Impact (ukr):**

Додавання нових параметрів у Desktop UI залишається помилконебезпечним:
поле може бути в catalog, але відсутнє на формі; бути на формі, але не
завантажуватись з preset; або бути неочевидним для користувача, бо залежність
від `mode`, `fusion_mode`, `learning_strategy`, `adapter` не відображена.

**Next direction:**

Introduce a `DesktopFieldRegistry` that normalizes catalog metadata into
`ui_level`, `tab`, `group`, `widget`, `active_when`, `default`,
`runtime_consumers`, and `derived_writes`. Continue the parallel PySide6
prototype in `tools/desktop_ui/`: first validate visual layout and field
coverage, then implement full load/save/run/status/log behavior. Keep composite
controls such as split-ratio editors and YAML blocks as explicit custom widgets.

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
