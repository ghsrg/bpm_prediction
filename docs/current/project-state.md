# Project State

Active project state for `bpm_prediction`.

This file is current-state documentation for agents and developers. It should be
shorter than canonical architecture documents and should not duplicate all
details from `README.MD`.

---

## Metadata

- `status`: active
- `audience`: human-and-agent
- `source_of_truth`: true
- `language_policy`: keys and section headers in English, human descriptions in Ukrainian
- `last_updated`: 2026-04-27
- `active_phase`: MVP2.5 Stage 4.2
- `primary_interface`: CLI

---

## Current State

- `mvp`: MVP2.5
- `stage`: Stage 4.2
- `runtime_status`: implemented
- `documentation_status`: canonical sync in progress

**Description (ukr):**

Проєкт зараз перебуває у фазі MVP2.5 Stage 4.2. Основний runtime вже
реалізований: є offline topology ingestion/sync, offline stats snapshots,
repository-backed train/eval path і Stage 4.2 stats integration. Основна
поточна робота по документації — зробити knowledge base керованою для агентів і
зафіксувати архітектурні рішення через ADR.

---

## Implemented Capabilities

### offline_topology_preparation

- `status`: implemented
- `commands`:
  - `ingest-topology`
  - `sync-topology`

**Description (ukr):**

Топологія процесів готується offline. Для одиничного dataset використовується
`ingest-topology`, для bulk-синхронізації — `sync-topology`. Runtime train/eval
не має перебудовувати topology з raw sources.

### offline_stats_snapshots

- `status`: implemented
- `commands`:
  - `sync-stats`
  - `sync-stats-backfill`
- `snapshot_policy`: immutable_json_asof

**Description (ukr):**

Статистика процесів готується offline і зберігається як immutable snapshots.
`sync-stats-backfill` використовується для побудови історичної timeline
snapshots, необхідної для `strict_asof` досліджень.

### repository_backed_runtime

- `status`: implemented
- `port`: `IKnowledgeGraphPort`
- `backends`:
  - `in_memory`
  - `file`
  - `neo4j`

**Description (ukr):**

Train/eval runtime споживає structure і stats через `IKnowledgeGraphPort`.
Backend має бути config-driven: перемикання між `file`, `in_memory`, `neo4j`
не повинно вимагати переписування domain/application логіки.

### stats_contract_v1

- `status`: implemented
- `contract_version`: `1.0`
- `producer`: `sync-stats`
- `consumer`: `DynamicGraphBuilder`
- `alignment_profiles`: `legacy_exact`, `safe_normalized`, `research_strict`
- `alignment_service`: `src/domain/services/activity_topology_alignment_service.py`

**Description (ukr):**

Universal stats contract v1 реалізовано: producer-side quality/alignment gates,
`metadata.stats_index`, `metadata.stats_contract`, consumer quality gate і
config-driven mapping у `struct_x`.

### gnn_runtime

- `status`: implemented
- `model_families`:
  - `BaselineGATv2`
  - `BaselineGCN`
  - `EOPKGGATv2`
  - `EOPKGGCN`

**Description (ukr):**

Підтримуються baseline GNN і EOPKG-моделі. EOPKG runtime може використовувати
`allowed_target_mask`, `structural_edge_index`, `structural_edge_weight`,
`struct_x` і snapshot telemetry.

### graph_dataset_cache_and_spill

- `status`: implemented
- `owner`: `src/cli.py`
- `keys`:
  - `experiment.graph_dataset_cache_policy`
  - `experiment.graph_dataset_cache_dir`
  - `experiment.graph_dataset_disk_spill_enabled`
  - `experiment.graph_dataset_shard_size`
  - `experiment.max_ram_gb`

**Description (ukr):**

Train/eval runtime має disk-cache для побудованих graph datasets і режим
sharded disk spill для великих запусків. Якщо spill увімкнено, runtime примусово
потребує cache write mode. `max_ram_gb` є soft RSS guard: при перевищенні ліміту
буфер графів flushиться у shards.

---

## Runtime Invariants

### pipeline_separation

- `adr`: `docs/adr/0002-offline-topology-runtime-separation.md`
- `rule`: train/eval/infer must consume prebuilt repository artifacts

**Description (ukr):**

Offline preparation і runtime ML фізично розділені. Це ключовий anti-leakage
інваріант MVP2.5.

### immutable_stats_snapshots

- `adr`: `docs/adr/0003-immutable-json-stats-snapshots.md`
- `rule`: snapshots are append-only, JSON-only, no TTL

**Description (ukr):**

Stats snapshots не перезаписуються і не нормалізуються в окремі `NodeStat` /
`EdgeStat` graph nodes. Поточний контракт — JSON payload + `metadata.stats_index`.

### strict_asof_research_policy

- `adr`: `docs/adr/0004-strict-asof-research-policy.md`
- `recommended_research_config`:
  - `experiment.stats_time_policy: strict_asof`
  - `experiment.on_missing_asof_snapshot: raise`

**Description (ukr):**

Для доказових temporal/drift експериментів треба використовувати `strict_asof`
і fail-fast поведінку при відсутньому snapshot. Fallback режими допустимі для
exploratory runs, але не для фінальних висновків.

---

### topology_projection_alignment

- `adr`: `docs/adr/0007-topology-projection-alignment.md`
- `config_keys`:
  - `mapping.graph_feature_mapping.topology_projection.gateway_mode`
  - `mapping.graph_feature_mapping.topology_projection.diagnostics_enabled`
  - `mapping.graph_feature_mapping.topology_projection.on_fail`
- `runtime_logs`: `Forward stats [...] topology_projection_*`

**Description (ukr):**

`collapse_for_prediction` has runtime diagnostics for projected topology:
alignment flag, projected edges/source paths, skipped edges, missing vocab,
duplicate labels, and missing node metadata. In train/eval/eval_drift these
counters are visible in `Forward stats [...]` logs, so agents and users can see
whether structural mapping was clean.

Recommended research-grade config:

```yaml
mapping:
  graph_feature_mapping:
    topology_projection:
      gateway_mode: collapse_for_prediction
      diagnostics_enabled: true
      on_fail: raise
```

---

## Runtime Semantics

### snapshot_creation_time

- `sync_stats_with_as_of`: uses explicit ISO timestamp
- `sync_stats_without_as_of`: derives `effective_as_of=max(event_ts)` after selected train-cut

**Description (ukr):**

Якщо `sync-stats` запускається з `--as-of`, snapshot отримує явно заданий час.
Без `--as-of` timestamp береться не як поточний UTC, а з даних: максимальний
`event_ts` після train-cut selection.

### snapshot_lookup_time

- `latest`: load latest structure/snapshot
- `strict_asof`: use prefix last-event timestamp

**Description (ukr):**

У `latest` режимі runtime бере останній доступний snapshot. У `strict_asof`
режимі lookup виконується на timestamp останньої події префіксу.

### missing_snapshot_policy

- `default_runtime_policy`: `disable_stats`
- `research_grade_policy`: `raise`
- `allowed_values`:
  - `disable_stats`
  - `use_base`
  - `raise`

**Description (ukr):**

Поточний runtime може не падати при missing/degraded snapshot і вимикати stats
branch. Для research-grade temporal runs потрібно використовувати `raise`.

---

## Baseline Compatibility

- `contract`: MVP1 baseline must stay green
- `stable_path`: `XESAdapter -> RawTrace -> PrefixPolicy -> PrefixSlice -> GraphBuilder -> GraphTensorContract -> Trainer`

**Description (ukr):**

MVP1 baseline path не можна ламати. Structural/EOPKG поля мають бути additive:
відсутність optional MVP2.5 tensors не повинна руйнувати baseline training.

---

## Config Ownership

### data_mapping

- `owners`: `data`, `mapping`
- `purpose`: source, adapter, backend, field mapping

### experiment

- `owners`: `experiment`
- `purpose`: run mode, split strategy, timing, temporal policies

### graph_feature_mapping

- `owners`: `mapping.graph_feature_mapping`
- `purpose`: stats-to-tensor mapping, node metrics, edge weights, encodings

### graph_dataset_cache

- `owners`: `experiment`
- `keys`:
  - `graph_dataset_cache_policy`
  - `graph_dataset_cache_dir`
  - `graph_dataset_disk_spill_enabled`
  - `graph_dataset_shard_size`
  - `max_ram_gb`
- `purpose`: graph dataset disk cache, sharded spill, and memory-bounded build_graph runs

### quality_policy

- `owners`:
  - `sync_stats.quality_gate`
  - `sync_stats.alignment_gate`
  - `mapping.graph_feature_mapping.stats_quality_gate`
- `purpose`: producer and consumer quality/alignment behavior

**Description (ukr):**

Нові config keys мають мати чітке місце. Якщо додається новий ключ у
`experiment`, `data`, `mapping`, `model`, `training`, `tracking` або
`sync_stats`, треба оновити `configs/ui/config_catalog.yaml`.

---

## Interfaces

### primary_cli

- `train_eval`: `.\.venv\Scripts\python.exe main.py --config <experiment.yaml>`
- `ingest_topology`: `.\.venv\Scripts\python.exe main.py ingest-topology --config <config.yaml> --split train --out <summary.json>`
- `sync_topology`: `.\.venv\Scripts\python.exe main.py sync-topology --config <config.yaml> --out <summary.json>`
- `sync_stats`: `.\.venv\Scripts\python.exe main.py sync-stats --config <config.yaml> --out <summary.json>`
- `sync_stats_asof`: `.\.venv\Scripts\python.exe main.py sync-stats --config <config.yaml> --as-of <ISO_TS> --out <summary.json>`
- `sync_stats_backfill`: `.\.venv\Scripts\python.exe main.py sync-stats-backfill --config <config.yaml> --step weekly --out-dir <dir>`
- `visualize_topology`: `.\.venv\Scripts\python.exe main.py visualize-topology --config <config.yaml> --version <version> --out <image.png>`
- `visualize_graph`: `.\.venv\Scripts\python.exe main.py visualize-graph --config <config.yaml> --pick latest --out <image.png>`
- `cache_clean`: `.\.venv\Scripts\python.exe main.py cache-clean --cache-dir .cache/graph_datasets --dry-run`
- `add_version2xes`: `.\.venv\Scripts\python.exe main.py add-version2xes --config <tool.yaml>`
- `simulate_versioned_log`: `.\.venv\Scripts\python.exe main.py simulate-versioned-log --config <tool.yaml>`

**Description (ukr):**

CLI є основним інтерфейсом запуску. `experiment-ui` — convenience wrapper,
`web-ui` — prototype, не primary runtime surface.

### verified_audit_points

- `main_py_router`: `main.py` routes operational subcommands to tool modules.
- `architecture_guard`: `.\.venv\Scripts\python.exe tools\architecture_guard.py`
  passed on 2026-04-27 with `[ARCH_GUARD] OK`.
- `cli_size`: `src/cli.py` is 2273 lines as of 2026-04-27; this confirms
  `cli_composition_root_overgrowth` debt.
- `sandbox_note`: plain sandbox execution of `.venv\Scripts\python.exe` may fail
  with base interpreter delegation to `AppData`; escalation can be required in
  Codex sessions.

**Description (ukr):**

Ці факти були звірені з кодом під час документаційного audit. Вони не є новими
runtime requirements, але допомагають наступним агентам відрізнити фактичний
стан від історичних worklogs.

---

## Validation Gates

Use project venv:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Description (ukr):**

У Codex sandbox plain `python` або `py -3` можуть не працювати. Канонічно
використовувати `.\.venv\Scripts\python.exe`. Якщо sandbox блокує запуск base
interpreter з AppData, треба повторити той самий venv command з escalation.

---

## Current Priorities

1. `documentation_governance`
   - keep `AGENTS.MD`, `docs/current/*`, and `docs/adr/*` aligned.
2. `research_grade_debt`
   - remaining active P0 debt is tracked in `docs/current/architecture-debt.md`;
     producer-side activity-to-topology alignment gate is closed.
3. `canonical_doc_sync`
   - remove stale "next step" wording from old canonical docs when touched.
4. `cli_primary_surface`
   - keep CLI as the primary run interface.

**Description (ukr):**

Поточний фокус — не додавати нові великі runtime фічі без потреби, а стабілізувати
документацію, ADR-рішення і P0 техборг, який блокує доказові експерименти.

---

## Related Current Docs

- `AGENTS.MD`
- `docs/current/architecture-debt.md`
- `docs/adr/README.md`
- `docs/ARCHITECTURE_MVP2_5.MD`
- `docs/DATA_MODEL_MVP2_5.MD`
- `docs/DATA_FLOWS_MVP2_5.MD`
- `docs/LLD_MVP2_5.MD`
- `docs/EVF_MVP2_5.MD`
- `docs/GNN_RUNTIME_MVP2_5.MD`

---

## Maintenance Rule

When current runtime behavior changes:

1. update this file,
2. update related ADR if an architectural decision changed,
3. update `docs/current/architecture-debt.md` if debt changed,
4. update `AGENTS.MD` only if routing or hard rules changed.
