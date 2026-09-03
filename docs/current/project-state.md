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
- `last_updated`: 2026-05-26
- `active_phase`: MVP2.5 Stage 4.2
- `primary_interface`: CLI

---

## Current State

- `mvp`: MVP2.5
- `stage`: Stage 4.2
- `runtime_status`: implemented
- `documentation_status`: canonical sync in progress

## Active Research Path

- `main_model_family`: `EOPKGTopologyConditioned`
- `candidate_contract`: `training.candidate_contract_mode=candidate_id`
- `candidate_identity`: `training.candidate_identity_mode=topology_native`
- `topology_conditioning`: `model.topology_conditioning_mode=impulse_activation_routing`
- `primary_output`: version-specific `candidate_logits [B, C_v]`
- `compatibility_output`: fixed-vocabulary diagnostics under `fixed_label_*`

**Summary:**

The current research path focuses on dynamic topology-native candidate scoring.
For each process version `v`, the topology defines the candidate space `C_v`.
The observed prefix is encoded, prefix state is injected as an impulse into the
current topology candidate graph, and the model ranks candidates in `C_v`.

Historical `EOPKGGATv2.fusion_mode` variants remain supported as experimental
controls and ablations. They should be treated as additional experiment modes,
not as the primary path for topology-native zero-shot adaptation.

## Current Research Hypothesis

- `topic`: topology-conditioned zero-shot predictive monitoring under structural process drift
- `dissertation_theme`: methodology of learning for optimizing business process management
- `business_assumption`: a new process topology/version can be available before enough event logs exist for model training
- `target_direction`: one fusion mode with a dedicated topology-conditioned training methodology, not a mixture of different fusion modes in one model
- `learning_strategy_doc`: `docs/GNN_LEARNING_STRATEGY.MD`
- `learning_methodology_doc`: `docs/GNN_LEARNING_METODOLOGY.MD`

**Hypothesis (ukr):**

Я бачу старі версії процесу та їхні логи. Модель має навчитися розуміти, як
topology впливає на `next-activity prediction`. Коли приходить нова topology
`vN`, навіть без логів `vN` для навчання, модель має використати цю topology як
умову прогнозу.

**Research framing (ukr):**

Поточні `fusion_mode` експерименти треба трактувати як ablation baseline: вони
показують, що проста подача структури може зменшувати OOS/ECE або давати
локальні покращення, але не дає стабільної значущої переваги після structural
drift. Основна дисертаційна цінність має бути в methodology: як навчити одну
topology-conditioned модель реально залежати від структури так, щоб нова BPMN /
EOPKG topology давала користь у cold-start/zero-shot режимі нової версії.

**Description (ukr):**

Проєкт зараз перебуває у фазі MVP2.5 Stage 4.2. Основний runtime вже
реалізований: є offline topology ingestion/sync, offline stats snapshots,
repository-backed train/eval path і Stage 4.2 stats integration. Основна
поточна робота по документації — зробити knowledge base керованою для агентів і
зафіксувати архітектурні рішення через ADR.

---

## Implemented Capabilities

### cdlg_benchmark_sequential_runner

- `status`: implemented in the current working tree; closure verification pending
- `entrypoint`: `tools/run_cdlg_benchmark.py`
- `plan`: `configs/ui/cdlg_benchmark_plan.yaml`
- `artifacts`: generated configs, per-run logs, and `manifest.jsonl` under `outputs/cdlg_benchmark/`

**Description (ukr):**

Для CDLG benchmark є headless runner, який виконує лише явно перелічені
пресети послідовно. Він показує в консолі номер запуску, назву preset-а,
поточний stage, completed/remaining у черзі та ETA на основі structured
progress events. Перед запуском рекомендовано виконати `--dry-run`.

When `experiment.statistic_enabled=false`, the runtime keeps topology-based
graph construction but does not resolve strict-as-of statistics snapshots or
emit missing-snapshot warnings.

### article_metric_export

- `status`: implemented
- `entrypoint`: `tools/export_mlflow_run_metrics_for_article.py`
- `selectors`:
  - `--runs-id`
  - `--runs-file`
  - `--experiment-id`
- `output_layout`: `learn/` and `drift/` metric CSV directories

**Description (ukr):**

Article metric export can discover active `FINISHED` MLflow runs by experiment
id while preserving the existing explicit run-id and run-file paths. Experiment
discovery uses the MLflow tracking root plus a separate experiment id, filters
out incomplete/deleted runs, and routes eligible runs into the existing
`learn/` or `drift/` directories from `experiment.mode`. The CSV contract keeps
existing fields and adds metadata-only `dataset_complexity`, read only from
explicit run params/tags.

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
  - `LSTM_Baseline`
  - `EOPKGGATv2`
  - `EOPKGGCN`
  - `EOPKGTopologyConditioned`

- `metric_space_status`: primary `strict_*`, `test_ece`, and `test_set_nll`
  use stable activity label/id semantics; frozen-vocabulary compatibility
  diagnostics use `fixed_label_*`.

- `research_role_groups`:
  - `primary_path`: `EOPKGTopologyConditioned` with `candidate_id`,
    `topology_native`, and `impulse_activation_routing`.
  - `reviewer_controls`: `LSTM_Baseline`, `BaselineGATv2`, `BaselineGCN`,
    `GATv2 + Mask`.
  - `experimental_structural_modes`: `EOPKGGATv2` fusion modes such as
    `ClassMeanAttention`, `ClassMeanConcat`, `ClassAwareStructuralScoring`,
    `TopologyStateEncoder`, `TopologyStateGraphEncoder`,
    `StructuralPriorEncoder`, and `StructXAttn`.

**Description (ukr):**

Підтримуються baseline GNN, fixed-vocabulary recurrent baseline і EOPKG-моделі. EOPKG runtime може використовувати
`allowed_target_mask`, `structural_edge_index`, `structural_edge_weight`,
`struct_node_to_class_index`, `struct_x` і snapshot telemetry. Для
`EOPKGGATv2` доступні backward-compatible `ClassMeanAttention` /
`ClassMeanConcat` і новий `ClassAwareStructuralScoring`, який додає
node-level structural logits, агреговані у class logits через
`struct_node_to_class_index`.

`TopologyStateEncoder` is available as an experimental `model.fusion_mode` for
early/input-level structural fusion. It consumes `struct_prefix_state_x`
(`[B, |V|, 6]` after batching) and projects prefix execution state onto
structural nodes before the structural GNN. This mode is intended as an
ablation for fusion-level and structural-overfitting analysis, not as the
canonical final drift-generalization mechanism.

Current `ClassAwareStructuralScoring` uses structural identity embeddings
enriched by `model.structural_stats_beta * stats_projection(struct_x)`, then a
bilinear prefix-to-structure scorer with a structural prior, late node-to-class
`LogSumExp` pooling, per-sample LayerNorm, and observed-logit scale alignment.
Trainer can add a set-aware structural auxiliary loss via
`training.structural_aux_loss_enabled` so the structural branch receives a
direct gradient signal.

`docs/GNN_LEARNING_STRATEGY.MD` defines the separation between `fusion_mode`,
`training.learning_strategy`, and `experiment.mode`.

`EOPKGTopologyConditioned` now supports a true `candidate_id` trainer path with
topology-homogeneous batching and candidate-level topology-flow diagnostics.
Optional `training.topology_flow_penalty_*` settings penalize probability mass
on candidates disallowed by the current topology flow. This targets the current
root-cause hypothesis: the model can see the candidate set, but may still ignore
changed BPMN/EOPKG edges unless the training objective explicitly pressures
flow-valid ranking.
`docs/GNN_LEARNING_METODOLOGY.MD` defines the research roadmap beyond current
fixed-head fusion ablations: topology-conditioned candidate scoring,
semantic-topological candidate prototypes, calibration/audit requirements, and
the staged path toward business-valid zero-shot adaptation.
`training.learning_strategy=standard` preserves current behavior.

`LSTM_Baseline` is available as a fixed-vocabulary logs-only recurrent
comparison model. It uses event feature embeddings plus numeric features,
LSTM/GRU sequence encoding, `last_node` or `global_mean` readout, and a linear
softmax head over `C_train`. It does not consume structural tensors and is used
to evaluate the static-vocabulary limitation under structural drift.
`GATv2 + Mask` is available as a fixed-vocabulary reviewer-control baseline:
it uses `BaselineGATv2` with inference-time hard topology post-filtering, while
retaining the static `C_train` output head.
For MLflow params, `BaselineGATv2` with `training.mask_guided_enabled=true`
is reported as `BaselineGATv2Mask` in both `model.type` and `model_type`, while
`model.base_type=BaselineGATv2` preserves the actual factory model identity.
`training.learning_strategy=topology_conditioned` is implemented as a
trainer-level methodology with known-version wrong-topology negatives,
same-version physical `drop_edges`, train-time allowed-set loss, and
version-weighted CE/rehearsal retention. `experiment.mode=versioned_zero_shot`
remains a separate future orchestration step.
Version-aware experiment sampling is available through
`experiment.fraction_strategy=versioned`, `experiment.split_strategy=versioned`,
and `experiment.version_scope_policy=train_cut`. This supports small balanced
known-version training samples without leaking future versions into train.
The desktop Experiment UI exposes these controls in the Core parameter group so
version-aware sampling can be configured without editing raw YAML.

`EOPKGTopologyConditioned` is available as an experimental `model.type` for
topology-conditioned candidate scoring. It is not an `EOPKGGATv2.fusion_mode`.
Stage 1 keeps the default trainer contract by returning fixed label logits
`[B, C_train]`, but internally scores topology nodes `[B, |V|]`, filters
`struct_node_to_class_index == -1`, and pools duplicate activity nodes to class
scores through MIL. `structural_edge_index` and `struct_node_to_class_index`
are required; `struct_x` is optional enrichment and the model can start from
candidate identity embeddings when `struct_x` is absent. It logs candidate
scale, entropy, duplicate-count, and target-vs-predicted score-gap diagnostics
for MLflow and selective traces. Stage 2 foundation adds
`forward_candidate(contract)` and `CandidatePredictionOutput` with dynamic
candidate logits `[B, C_v]`, `candidate_class_index`, `node_logits`, and
`node_to_candidate_index`. With
`training.dynamic_candidate_contract_enabled=true` or
`training.candidate_contract_mode=fixed_projection`, train/eval/drift consume
`forward_candidate()` and project topology-local candidate logits back to sparse
fixed-label logits for compatibility with existing CE, mask, calibration, and
drift metrics.

`training.candidate_contract_mode=candidate_id` now enables the first true
candidate-level trainer/evaluator path: training uses set-aware CE over
topology-local `candidate_logits [B, C_v]`, where all topology candidates mapped
to the target activity class are accepted as the target set. Evaluation and
one-pass drift inference still map candidate probabilities/predictions back to
global activity classes for the existing metric pipeline, but the loss no longer
requires a fixed `[B, C_train]` projection. Candidate target diagnostics report
`candidate_target_in_candidate_set_rate`, `candidate_missing_target_rate`,
duplicate target count, and target-set logit variance/entropy. Training fails
when `candidate_missing_target_rate` exceeds
`training.candidate_missing_target_fail_threshold` because that indicates broken
log/topology alignment.

For `candidate_id`, the DataLoader uses topology-homogeneous batching by
`process_version_idx + stats_snapshot_version_idx` for indexable graph sources,
including in-memory datasets and `ShardedGraphDataset` disk-cache sources. A
safety guard rejects mixed topology batches that still reach the trainer. New
sharded graph cache entries persist lightweight
`topology_segments` per shard, so candidate-id batching can form topology
groups without hydrating every structural payload before inference.

Run-Status structured progress includes `eval_drift.one_pass_inference` between
`build_graph.test` and `eval_drift.windows`, so long one-pass drift inference on
prebuilt test graphs is visible in the desktop Experiment UI.

Stats-backed structural drift runtime now uses snapshot-aware Neo4j stats
payload caching and deduplicated structural payload shards. Heavy stats payloads
are loaded by resolved snapshot identity instead of repeatedly loading full JSON
payloads for every exact prefix `as_of_ts`. Sharded graph cache files can store
one structural payload per repeated `structural_payload_key` and rehydrate it at
load time.

`DynamicGraphBuilder` DTO cache applies to `strict_asof` lookups when
`cache_policy` is `dto` or `full` and stats-backed `graph_feature_mapping` is
not active; repeated prefixes with the same version and timestamp do not
repeatedly call the knowledge repository. When stats-backed
`graph_feature_mapping.enabled=true`, DTO entries are intentionally not cached
per exact prefix `as_of_ts`; compiled topology/stat payloads are deduplicated by
resolved snapshot identity instead.

`eval_drift` can use one-pass prebuilt test dataset evaluation when graph
samples carry `trace_idx` metadata. Drift-window metrics are aggregated from
compact per-sample inference records instead of rebuilding graph windows or
re-running model forward for every overlapping window.
`test_top3_accuracy` uses an effective `k < n_classes` for small class spaces
and falls back to plain accuracy for binary/single-class windows, so small test
sets do not report mathematically meaningless perfect top-k diagnostics.

`tools/audit_topology_drift.py` is available as an offline diagnostic for
completed train/eval_drift runs. It compares loan BPMN source files through a
gateway-collapsed prediction view and writes audit artifacts under
`outputs/audits/topology_drift/`. Use it before claiming zero-shot structural
benefit when new/removed process candidates may explain drift degradation.
The audit can backfill `prefix_last_activity` for older trace artifacts from
XES via `(trace_idx, prefix_len)`, and new graph payloads carry
`prefix_last_activity_idx` so future traces expose changed-transition context
directly.

`StructuralPriorEncoder` is available as an etalon-like `model.fusion_mode`
for `EOPKGGATv2`. It keeps the observed prefix encoder as the primary path,
mean-pools the structural GNN node states into `struct_context`, and fuses
`[obs_context || struct_context]` before the classifier. It supports
`model.structural_prior_fusion=concat` and `gated_concat`; trainer diagnostics
log context scale and gate metrics.

`TopologyStateGraphEncoder` is available as an experimental
`model.fusion_mode` for `EOPKGGATv2`. It consumes `struct_prefix_state_x`,
projects observed prefix state onto structural nodes, runs structural message
passing over `structural_edge_index`, mean-pools the topology node states into
a graph-level context, and classifies that structural graph context. This mode
is the etalon/article-like structural graph baseline for same-version topology
usefulness checks, not the final drift-transfer mechanism.

`StructXAttn` is available as a token-level structural cross-attention
`model.fusion_mode` for `EOPKGGATv2`. It keeps the observed IG encoder active,
encodes structural nodes with the structural GNN, and lets observed nodes query
structural node states before pooling. It supports `model.struct_xattn_layers`
values `post_conv2` and `after_each_conv`. Trainer can optionally enable
correct-vs-corrupted topology training through
`training.struct_xattn_contrastive_enabled`; this objective is train-only and
logs StructXAttn contribution and corrupted-topology delta diagnostics.
StructXAttn residual fusion is configurable through
`model.struct_xattn_merge_mode`. Historical runs use `post_norm_residual`;
new drift-stabilization experiments should prefer `pre_norm_context` with an
optional `model.struct_xattn_delta_ratio_max` cap to prevent post-merge
LayerNorm amplification of structural deltas.

Selective structural prediction tracing is available through
`tracking.tracing.*`. It records a bounded set of explanation/debug traces for
interesting predictions during test or one-pass drift evaluation. MLflow 3
spans are used when available; otherwise a rank/PID-scoped JSONL fallback is
logged as an artifact. Trace payloads detach tensors before serialization and
store searchable flat attributes separately from nested explanation JSON.

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

For stats-backed structural runs, sharded cache uses a deduplicated shard
payload format: per-prefix `Data` objects store `structural_payload_key`, while
`struct_x`, `structural_edge_index`, `structural_edge_weight`, and
`struct_node_to_class_index` are stored once per shard payload key and reattached
by CLI diagnostics and `ShardedGraphDataset`.

Graph dataset cache schema `4` includes drift metadata and the optional
`struct_prefix_state_x` contract field. Old cache entries without `trace_idx`
fall back to legacy raw-trace drift evaluation; cache entries without
`struct_prefix_state_x` cannot run `model.fusion_mode=TopologyStateEncoder`.

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

### mlflow_model_logging_compatibility

- `port`: `ITracker.log_model(model, artifact_path)`
- `adapter`: `MLflowTracker`
- `mlflow_2_behavior`: call `mlflow.pytorch.log_model(model, artifact_path)`
- `mlflow_3_behavior`: call `mlflow.pytorch.log_model(model, name=artifact_path)`

**Description (ukr):**

Tracking adapter preserves the application-level `artifact_path` contract and
selects the compatible `mlflow.pytorch.log_model` signature at runtime. This
keeps the old `.venv` with MLflow 2.x and `.venv-modern` with MLflow 3.x
compatible without changing trainer/use-case code.

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
`experiment`, `data`, `mapping`, `model`, `training`, `tracking` Р°Р±Рѕ
`sync_stats`, треба оновити `configs/ui/config_catalog.yaml`.

---

## Interfaces

### primary_cli

- `train_eval`: `.\.venv-modern\Scripts\python.exe main.py --config <experiment.yaml>`
- `ingest_topology`: `.\.venv-modern\Scripts\python.exe main.py ingest-topology --config <config.yaml> --split train --out <summary.json>`
- `sync_topology`: `.\.venv-modern\Scripts\python.exe main.py sync-topology --config <config.yaml> --out <summary.json>`
- `sync_stats`: `.\.venv-modern\Scripts\python.exe main.py sync-stats --config <config.yaml> --out <summary.json>`
- `sync_stats_asof`: `.\.venv-modern\Scripts\python.exe main.py sync-stats --config <config.yaml> --as-of <ISO_TS> --out <summary.json>`
- `sync_stats_backfill`: `.\.venv-modern\Scripts\python.exe main.py sync-stats-backfill --config <config.yaml> --step weekly --out-dir <dir>`
- `visualize_topology`: `.\.venv-modern\Scripts\python.exe main.py visualize-topology --config <config.yaml> --version <version> --out <image.png>`
- `visualize_graph`: `.\.venv-modern\Scripts\python.exe main.py visualize-graph --config <config.yaml> --pick latest --out <image.png>`
- `cache_clean`: `.\.venv-modern\Scripts\python.exe main.py cache-clean --cache-dir .cache/graph_datasets --dry-run`
- `add_version2xes`: `.\.venv-modern\Scripts\python.exe main.py add-version2xes --config <tool.yaml>`
- `simulate_versioned_log`: `.\.venv-modern\Scripts\python.exe main.py simulate-versioned-log --config <tool.yaml>`

**Description (ukr):**

CLI is the primary execution interface. `experiment-ui` is the supported
legacy Tkinter desktop wrapper. `ui` is the parallel PySide6 desktop UI
prototype for catalog-driven Project Setup / Experiment Run / Advanced layout
review.

### verified_audit_points

- `main_py_router`: `main.py` routes operational subcommands to tool modules.
- `architecture_guard`: `.\.venv-modern\Scripts\python.exe tools\architecture_guard.py`
  passed on 2026-04-27 with `[ARCH_GUARD] OK`.
- `cli_size`: `src/cli.py` is 2273 lines as of 2026-04-27; this confirms
  `cli_composition_root_overgrowth` debt.
- `sandbox_note`: plain sandbox execution of `.venv-modern\Scripts\python.exe` may fail
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
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
.\.venv-modern\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv-modern\Scripts\python.exe -m pytest tests/ -v
```

**Description (ukr):**

У Codex sandbox plain `python` або `py -3` можуть не працювати. Канонічно
використовувати `.\.venv-modern\Scripts\python.exe`. Якщо sandbox блокує запуск base
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
- `docs/GNN_LEARNING_STRATEGY.MD`

---

## Maintenance Rule

When current runtime behavior changes:

1. update this file,
2. update related ADR if an architectural decision changed,
3. update `docs/current/architecture-debt.md` if debt changed,
4. update `AGENTS.MD` only if routing or hard rules changed.

---

## Runtime Update 2026-05-26

`EOPKGTopologyConditioned` candidate-id mode now has a topology-native candidate
identity path:

```yaml
training.candidate_contract_mode: candidate_id
training.candidate_identity_mode: topology_native
```

The graph contract can carry topology-local `candidate_ids`,
`candidate_labels`, `candidate_class_index`, `candidate_is_unseen`,
`struct_node_to_candidate_index`, `candidate_allowed_target_mask`, and raw
`target_label`.

This prevents future topology candidates from being removed only because they
are absent from the training activity vocabulary. Fixed-label metrics remain
compatibility diagnostics when candidates map to `candidate_class_index=-1`.

## Runtime Update 2026-05-27

Fixed a target-to-candidate alignment mismatch during `candidate_id` training with `topology_native` identity mode. Previously, raw target labels (usually node IDs like `t_approve_loan` from log `concept:name`) were compared only against human-readable `candidate_labels` (like `Approve Loan`), resulting in a 100% missing target rate and training failure. The target mapping helper `candidate_target_mask_from_labels` and the prediction model's `map_target_labels_to_candidate_mask` now match targets against both `candidate_ids` and `candidate_labels`.

## 2026-08-31 MOU-001 Topology-Mask Uniform Baseline

`MOU-001` is complete with closure evidence in
`outputs/worklogs/2026-08-31-closure-mou-001.md`. The slice adds
`experiment.mode=eval_topology_mask_uniform`, a non-neural reviewer baseline
that evaluates `candidate_allowed_target_mask [B, C_v]` directly. The evaluator
reports analytical `uniform_mask_expected_accuracy`, structural
`test_target_in_mask_rate`, mask cardinality/reduction diagnostics, seeded
Monte Carlo strict/hybrid F1 summaries, and scalar drift windows. It is
evaluation-only and must load only `encoder_state` from the configured reference
checkpoint. When MLflow tracking is enabled, this internal mode is recorded with
the standard `mode=eval_drift` tag and `params.experiment.mode=eval_drift` for
downstream run placement, while the original evaluator mode is retained in
`evaluation_mode` and `params.experiment.evaluation_mode`. MOU runs use
`model_type=MOU` / `model.type=MOU` and log drift-window scalar series using
the standard `drift_window_*` metric namespace so MLflow comparison plots can
overlay them with neural `eval_drift` runs. Drift windows use the same
`experiment.drift_window_size` / `experiment.drift_window_sliding` policy and
trace-axis drop-short-tail behavior as neural `eval_drift`, while each window
contains all prefix records belonging to the selected traces. Progress is
reported through the standard `eval_drift.one_pass_inference` and
`eval_drift.windows` stages.

## Runtime Update 2026-06-12

Extended topology-native candidate target matching for XES lifecycle-classified
labels. Candidate-id training now matches exact `candidate_ids` /
`candidate_labels` first, then deterministic aliases that strip known lifecycle
suffixes such as `+COMPLETE`. This fixes BPI-style targets like
`W_Nabellen offertes+COMPLETE` failing to match topology candidates like
`W_Nabellen offertes` and causing `candidate_missing_target_rate=1.0`.

## Runtime Update 2026-05-29

Optimized CPU and memory performance inside `ModelTrainer` during batched forward passes. Previously, structural and candidate payload resolution for the first graph in a batch fell back to `to_data_list()` or `get_example(0)`. This forced PyTorch Geometric to slice all batched tensors into individual sample `Data` objects on every step, creating a massive CPU serialization bottleneck. We replaced this with direct slice dictionary-based indexing using PyG's built-in `_slice_dict` pointers, which provides O(1) attribute extraction for both homogeneous and heterogeneous batches without materializing sample graphs.

Additionally resolved a major disk I/O loading bottleneck in `ShardedGraphDataset`. During training on full datasets with shuffled indexing, accessing graphs randomly across shards triggered severe cache thrashing when the total number of shards exceeded the default `max_cached_shards` (2). This forced PyTorch to reload 80MB+ shard files from disk repeatedly on almost every batch step, causing epoch times to swell to ~60 hours. We implemented automatic scaling of `max_cached_shards` to match the actual number of shards in the dataset split, and added a corresponding `training.sharded_dataset_cached_shards` configuration key to the config catalog. Shards are now loaded exactly once per epoch, yielding a speedup of >250x and restoring the training runtime back to baseline.

## Runtime Update 2026-05-30

Verified model retraining and F1 convergence using `configs/experiments/_EOPKGTC-UN-42_test_grounding.yaml` (fraction 0.2, 50 epochs). The training run successfully completed with gradient stability (loss stable at ~3.35, no explosions). Evaluation masking correctly filtered technical gateway candidates at metric-evaluation time. Validation F1 successfully converged (peaking at ~0.52 on val, with final test F1 reaching 0.67 and top-3 accuracy at 0.99). This confirms that both the convergence degradation (previously stuck at F1 ~0.43) and cache loading issues are resolved, and the pipeline is fully prepared for full-scale runs.

## Runtime Update 2026-05-31

`EOPKGTopologyConditioned` now supports experimental
`model.topology_conditioning_mode=impulse_activation_routing` for
topology-native candidate scoring. The mode keeps the observed prefix encoder,
requires `struct_prefix_state_x`, injects a normalized execution impulse into
candidate node states, propagates it through the current structural topology,
and returns `candidate_logits [B, K_v]` through `forward_candidate()`.

This is not an `EOPKGGATv2.fusion_mode` and does not make fixed-label logits the
primary loss surface. Recommended research runs use
`training.candidate_contract_mode=candidate_id` and
`training.candidate_identity_mode=topology_native`; fixed-label projection is
compatibility reporting only.

## Runtime Update 2026-06-02

`simulate-versioned-log` now supports bounded probabilistic branches for
exclusive-gateway loop scenarios through branch-level `probability`,
`max_traversals_per_case`, and `repeat_until_max_once_selected` settings.

This keeps old simulator configs reproducible because existing deterministic
`when` rules remain unchanged, while new configs can generate 5-10% repeated
loop traces without risking unbounded BPMN loop execution. Parallel gateways
continue to execute all outgoing branches and do not consume branch
probabilities.

Simulator runs now also write a dataset statistics artifact controlled by
`output.dataset_stats_json_path`. The JSON captures total and per-version trace
length, cycle depth, version carryover, task-node coverage, node usage
distribution, and resource/task distribution metrics for reproducibility audits.

`simulate-versioned-log` also supports case-level `version_carryover` targets.
When enabled, the simulator samples a desired completion bucket per case
(`same_version`, `next_version`, `skip_one_version`, `last_version`, or
explicit `plus_N`) and inserts waiting time before a terminal task when needed,
so generated XES event timestamps can cross process-version activation
boundaries in a controlled way.

For more organic long-running synthetic logs, task configs can use
`conditional_waits` evaluated from trace-level `case_attributes`. These waits
advance case time before a task starts without holding worker resources, making
it possible to model document delays, client-side waiting, replacement approval,
or audit queues across many tasks. Dataset statistics now report both
`version_carryover` and calendar-month `calendar_carryover` buckets.

Dataset statistics also report `bpms_native_operational_variance` instead of
data-corruption noise terminology. The section captures flattened parallel
interleaving, technical retries/incidents, and resource
substitution/delegation patterns that are native to BPMS-style event logs.

## Runtime Update 2026-06-02: Process-State-Aware Parallel Masks

`DynamicGraphBuilder` now supports optional process-state-aware mask expansion
for parallel/interleaved XES logs:

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: lifecycle_active_set
training.process_state_mask_include_direct_successors: true
training.process_state_mask_include_active_candidates: true
```

When enabled, the builder reconstructs active tasks from prefix lifecycle events
(`lifecycle:transition=start|complete`) keyed by `sim:activity_instance_id` and
merges still-active tasks into both `allowed_target_mask` and
`candidate_allowed_target_mask`. The existing `struct_prefix_state_x[:, 5]`
active-candidate channel is reused, so tensor shape remains `[|V|, 6]`.

When `training.process_state_mask_enabled=false`, lifecycle-active metadata on
events (`active_activities_after_complete` and
`active_activity_counts_after_complete`) is ignored. The metadata is also
ignored when `training.process_state_mask_include_active_candidates=false`.
This prevents oracle leakage from CDLG overlap metadata into class masks,
candidate masks, and `struct_prefix_state_x[:, 5]`.

This is intended for XES logs where parallel branch completions can appear after
another branch's completion. Completion-only logs cannot reliably reconstruct
the active set and should keep the default direct-successor behavior until a
relaxed reachability/window mask is implemented.

## Runtime Update 2026-06-02: Relaxed Reachability Parallel Mask

`DynamicGraphBuilder` now implements the planned
`training.process_state_mask_source=relaxed_reachability` fallback. The mode
uses recent completed prefix activities as anchors and adds bounded reachable
successors from the collapsed prediction topology:

```yaml
training.process_state_mask_enabled: true
training.process_state_mask_source: relaxed_reachability
training.process_state_mask_relaxed_lookback_events: 8
training.process_state_mask_relaxed_max_depth: 1
training.process_state_mask_relaxed_max_cardinality_ratio: 0.35
```

This is an approximate fallback for flattened parallel XES, not exact BPMN
marking replay. Runtime diagnostics now include mean active and relaxed
candidate counts through forward stats/MLflow metric prefixes.

## Runtime Update 2026-06-03: State-Aware Relaxed Reachability

`relaxed_reachability` now supports completed-candidate suppression,
open-successor anchor filtering, and active-instance protection:

```yaml
training.process_state_mask_relaxed_suppress_completed: true
training.process_state_mask_relaxed_anchor_policy: open_successors
training.process_state_mask_relaxed_loop_policy: keep_direct_successor_repeats
```

The completed filter applies only to relaxed-only candidates. Direct successors
of the last completed task, explicit active candidates, currently active task
tokens, and direct loop/rework repeats remain allowed. This prevents the relaxed
mask from keeping stale completed parallel siblings while avoiding false
negatives for multiple-instance or rework cases.

New diagnostics include:

- `*_process_state_mask_relaxed_raw_candidate_count`
- `*_process_state_mask_relaxed_suppressed_completed_count`
- `*_process_state_mask_relaxed_final_candidate_count`
- `*_process_state_mask_target_suppressed_by_completed_filter_rate`
- `*_process_state_mask_completed_suppression_rate`

For completion-only XES views over collapsed BPMN gateways, relaxed reachability
also preserves not-yet-completed initial candidates reachable from `startEvent`
through transparent gateways. This prevents valid parallel initial sibling
tasks from producing empty topology-native candidate masks during MOU or
mask-aware drift evaluation.

## Runtime Update 2026-09-02: Strict-As-Of Mask Topology Lookup

`DynamicGraphBuilder` now resolves the process structure DTO with
`get_process_structure_as_of()` whenever `experiment.stats_time_policy` is
`strict_asof`, using the last prefix event timestamp as the `as_of_ts`.

This applies even when stats-backed structural features are disabled. The
topology-native masks, including process-state and relaxed-reachability masks,
still depend on the temporal topology snapshot and must not silently fall back
to the latest structure.

## Runtime Update 2026-09-03: Fixed-Vocab BPMN Topology Bridge

`DynamicGraphBuilder` now supports BPMN node ids that differ from XES activity
labels when `training.candidate_identity_mode=fixed_vocab_bridge`. The
projection keeps BPMN ids for topology paths and edge-stat bookkeeping, then
maps prediction endpoints through unique DTO node metadata into the fitted
activity vocabulary before writing class-space `structural_edge_index`,
`allowed_target_mask`, and `candidate_allowed_target_mask`.

Unsafe bridge mappings are diagnostic-only skips: missing labels, labels absent
from `activity_vocab`, and duplicate prediction labels do not guess a class.
`topology_native` remains BPMN-node based. Graph dataset cache schema is `7`,
so pre-bridge graph caches are rejected and rebuilt.

## Runtime Update 2026-06-10: Impulse State Channel Config Guard

`model.impulse_state_channels` is now normalized by the CLI composition root
before model construction. List and YAML-list text values are preserved, while
empty or scalar UI values fall back to the default impulse channels:

```yaml
model.impulse_state_channels:
  - is_last_event
  - prefix_executed_count_log1p
  - prefix_recency_norm
```

This prevents `EOPKGTopologyConditioned` runs from failing at model
construction with `TypeError: 'int' object is not iterable` when an Experiment
UI preset accidentally stores the field as `0` or `1`.
