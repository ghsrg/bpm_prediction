# Finish_MVP2_5_Plan.MD

Date: 2026-03-20  
Status: Updated (with universal stats contract plan)

## 1. Поточний стан MVP2.5

Виконано:
1. Offline ingestion/sync topology (`ingest-topology`, `sync-topology`).
2. Offline stats snapshots (`sync-stats`) з immutable snapshot policy.
3. Repository-driven runtime path через `IKnowledgeGraphPort`.
4. Neo4j backend інтегровано та валідовано.
5. XES Stage4.2 дефект `mostly 0.0` локалізовано та виправлено для `BPI_Challenge_2012` (alignment policy).

Ще не закрито:
1. Універсальний контракт статистики (формально зафіксований і стабільний для producer/consumer).
2. Runtime quality gate у consumption path (`DynamicGraphBuilder`).
3. Config-driven stats mapping у GNN за `encoding` з `data/mapping` config.
4. Повний release checklist для MVP2.5 з quality acceptance.

## 2. Ключове рішення по порядку робіт

Порядок реалізації фіксуємо як обов'язковий:
1. Universal stats contract v1.
2. Runtime quality gate на основі contract quality fields.
3. Config-driven mapping у GNN через `encoding`.
4. Додавання в forward статиситку в структуру

Причина:
1. `encoding`-мапінг має спиратися на стабільні ключі метрик і quality semantics.
2. Без contract-first буде подвійний рефакторинг builder/feature bridge.

## 3. Пріоритети завершення MVP2.5

`P0` (блокери закриття MVP2.5):
1. Реалізувати universal stats contract v1.
2. Реалізувати runtime quality gate у consumer path.
3. Зафіксувати acceptance thresholds і warning/fallback policy.
4. Додати E2E тести (aligned/non-aligned, strict_asof anti-leakage, zero-dominant fallback).

`P1` (високий пріоритет, бажано до freeze):
1. Config-driven GNN stats feature mapping за `encoding`.
2. Cross-backend parity tests (`file` vs `neo4j`) для stats payload semantics.
3. Короткий quality summary у CLI-output і run logs.

`P2` (можна в MVP3 backlog):
1. Автоматичний discovery mapping між XES/BPMN.
2. Adaptive policy selection (`use_classifier`, normalization rules) per dataset.
3. Розширена observability-аналітика quality drift.

## 4. План реалізації universal stats contract v1

## 4.1 Scope контракту

Контракт покриває:
1. Producer (`sync-stats`) - що і як зберігається в snapshot JSON.
2. Consumer (`DynamicGraphBuilder`) - як читається і перевіряється quality.
3. Feature mapping layer - як метрики підтягуються у GNN фічі за config.

## 4.2 Contract schema (v1, proposal)

Обов'язкові блоки у snapshot metadata/payload:
1. `stats_contract_version: "1.0"`
2. `stats_identity`:
   - `tenant_id`
   - `process_name`
   - `version_key`
   - `proc_def_id`
   - `knowledge_version`
   - `as_of_ts`
3. `stats_quality`:
   - `history_coverage_percent`
   - `non_zero_ratio_node`
   - `non_zero_ratio_edge`
   - `non_zero_ratio_global`
   - `zero_dominant` (bool)
   - `is_usable_for_training` (bool)
   - `quality_reason` (enum string)
4. `stats_policy`:
   - `stats_time_policy`
   - `process_scope_policy`
   - `split_strategy`
   - `train_ratio`
   - `fraction`
5. `stats_index` (canonical access map):
   - `node`: key -> `{entity_id: value}`
   - `edge`: key -> `{edge_id_or_pair: value}`
   - `global`: key -> `value`

Canonical key format:
1. `{window}.{scope}.{metric}`
2. Example: `all_time.version.exec_count`

## 4.3 Runtime quality gate (contract-driven)

Місце: `DynamicGraphBuilder` consumption path.

Правила v1:
1. Якщо snapshot відсутній -> stats tensors не додаються, warning.
2. Якщо `is_usable_for_training=false` -> stats tensors не додаються, warning.
3. Якщо `zero_dominant=true` або `non_zero_ratio_global < threshold` -> ignore stats tensors, warning.
4. Model path не падає, працює без stats branch.

Config knobs:
1. `runtime.stats_quality_gate.enabled: true`
2. `runtime.stats_quality_gate.zero_dominant_threshold: 0.95`
3. `runtime.stats_quality_gate.min_non_zero_ratio_global: 0.05`
4. `runtime.stats_quality_gate.on_fail: "ignore_with_warning"`

## 4.4 Config-driven mapping через `encoding`

Мета:
1. Прибрати hardcoded baseline aggregation у GNN.
2. Брати stats features з `mapping.graph_feature_mapping` + `features[*].encoding`.

Правила:
1. Кожна stats feature описується у config як:
   - `name`
   - `metric`
   - `window`
   - `scope`
   - `default`
   - `encoding` (наприклад `["z-score"]`, `["identity"]`, `["log1p"]`)
2. Builder використовує лише canonical `stats_index` ключі.
3. Якщо ключ відсутній або quality gate забороняє використання - підставляється `default`.

## 4.5 Implementation steps

Step 1:
1. Додати `stats_contract_version` і `stats_quality` у `sync-stats` payload.
2. Уніфікувати формування canonical `stats_index` ключів.

Step 2:
1. Додати policy-функцію оцінки quality (`zero_dominant`, `is_usable_for_training`, `quality_reason`).
2. Записувати quality у snapshot та summary.

Step 3:
1. Додати contract-reader у `DynamicGraphBuilder`.
2. Реалізувати runtime quality gate.

Step 4:
1. Впровадити config-driven mapping stats->tensor за `encoding`.
2. Залишити backward-compatible fallback для існуючих config.

Step 5:
1. Оновити docs (`DATA_MODEL_MVP2_5`, `DATA_FLOWS_MVP2_5`, runbook section).
2. Додати release note з known limitations.

## 4.6 Test plan

Unit tests:
1. `sync-stats` формує коректний `stats_contract_version` і `stats_quality`.
2. Quality policy правильно класифікує zero-dominant/non-zero snapshots.
3. Builder fallback path без stats tensors працює стабільно.

Integration tests:
1. XES aligned case -> `is_usable_for_training=true`, non-zero stats tensors.
2. XES non-aligned case -> `is_usable_for_training=false`, tensors ignored.
3. strict_asof + train-cut -> без leakage, правильний `as_of`.

Regression gates:
1. `py -3 tools/architecture_guard.py`
2. `pytest -m mvp1_regression -v`
3. `pytest tests/ -v`

## 5. Що критично важливо з техборгу

Найкритичніше виправити до MVP3:
1. Runtime quality gate (`DEBT-A`) - інакше noise потрапляє в GNN.
2. Snapshot quality contract (`DEBT-C`) - інакше немає надійної source-agnostic поведінки.
3. Mapping bridge policy (`DEBT-B`) - інакше mixed-source сценарії нестабільні.

## 6. Критерії завершення MVP2.5

MVP2.5 закриваємо, якщо:
1. P0 завершено і протестовано.
2. Runtime path не споживає zero-dominant stats.
3. Config-driven mapping працює через canonical contract keys.
4. Усі обов'язкові test gates зелені.
5. Є фінальний release report + перенесений MVP3 backlog.
