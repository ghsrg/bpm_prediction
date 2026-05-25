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
- `last_updated`: 2026-05-25
- `active_phase`: MVP2.5 Stage 4.2
- `primary_interface`: CLI

---

## Current State

- `mvp`: MVP2.5
- `stage`: Stage 4.2
- `runtime_status`: implemented
- `documentation_status`: canonical sync in progress

## Current Research Hypothesis

- `topic`: topology-conditioned zero-shot predictive monitoring under structural process drift
- `dissertation_theme`: methodology of learning for optimizing business process management
- `business_assumption`: a new process topology/version can be available before enough event logs exist for model training
- `target_direction`: one fusion mode with a dedicated topology-conditioned training methodology, not a mixture of different fusion modes in one model
- `learning_strategy_doc`: `docs/GNN_LEARNING_STRATEGY.MD`
- `learning_methodology_doc`: `docs/GNN_LEARNING_METODOLOGY.MD`

**Hypothesis (ukr):**

РЇ Р±Р°С‡Сѓ СЃС‚Р°СЂС– РІРµСЂСЃС–С— РїСЂРѕС†РµСЃСѓ С‚Р° С—С…РЅС– Р»РѕРіРё. РњРѕРґРµР»СЊ РјР°С” РЅР°РІС‡РёС‚РёСЃСЏ СЂРѕР·СѓРјС–С‚Рё, СЏРє
topology РІРїР»РёРІР°С” РЅР° `next-activity prediction`. РљРѕР»Рё РїСЂРёС…РѕРґРёС‚СЊ РЅРѕРІР° topology
`vN`, РЅР°РІС–С‚СЊ Р±РµР· Р»РѕРіС–РІ `vN` РґР»СЏ РЅР°РІС‡Р°РЅРЅСЏ, РјРѕРґРµР»СЊ РјР°С” РІРёРєРѕСЂРёСЃС‚Р°С‚Рё С†СЋ topology СЏРє
СѓРјРѕРІСѓ РїСЂРѕРіРЅРѕР·Сѓ.

**Research framing (ukr):**

РџРѕС‚РѕС‡РЅС– `fusion_mode` РµРєСЃРїРµСЂРёРјРµРЅС‚Рё С‚СЂРµР±Р° С‚СЂР°РєС‚СѓРІР°С‚Рё СЏРє ablation baseline: РІРѕРЅРё
РїРѕРєР°Р·СѓСЋС‚СЊ, С‰Рѕ РїСЂРѕСЃС‚Р° РїРѕРґР°С‡Р° СЃС‚СЂСѓРєС‚СѓСЂРё РјРѕР¶Рµ Р·РјРµРЅС€СѓРІР°С‚Рё OOS/ECE Р°Р±Рѕ РґР°РІР°С‚Рё
Р»РѕРєР°Р»СЊРЅС– РїРѕРєСЂР°С‰РµРЅРЅСЏ, Р°Р»Рµ РЅРµ РґР°С” СЃС‚Р°Р±С–Р»СЊРЅРѕС— Р·РЅР°С‡СѓС‰РѕС— РїРµСЂРµРІР°РіРё РїС–СЃР»СЏ structural
drift. РћСЃРЅРѕРІРЅР° РґРёСЃРµСЂС‚Р°С†С–Р№РЅР° С†С–РЅРЅС–СЃС‚СЊ РјР°С” Р±СѓС‚Рё РІ methodology: СЏРє РЅР°РІС‡РёС‚Рё РѕРґРЅСѓ
topology-conditioned РјРѕРґРµР»СЊ СЂРµР°Р»СЊРЅРѕ Р·Р°Р»РµР¶Р°С‚Рё РІС–Рґ СЃС‚СЂСѓРєС‚СѓСЂРё С‚Р°Рє, С‰РѕР± РЅРѕРІР° BPMN /
EOPKG topology РґР°РІР°Р»Р° РєРѕСЂРёСЃС‚СЊ Сѓ cold-start/zero-shot СЂРµР¶РёРјС– РЅРѕРІРѕС— РІРµСЂСЃС–С—.

**Description (ukr):**

РџСЂРѕС”РєС‚ Р·Р°СЂР°Р· РїРµСЂРµР±СѓРІР°С” Сѓ С„Р°Р·С– MVP2.5 Stage 4.2. РћСЃРЅРѕРІРЅРёР№ runtime РІР¶Рµ
СЂРµР°Р»С–Р·РѕРІР°РЅРёР№: С” offline topology ingestion/sync, offline stats snapshots,
repository-backed train/eval path С– Stage 4.2 stats integration. РћСЃРЅРѕРІРЅР°
РїРѕС‚РѕС‡РЅР° СЂРѕР±РѕС‚Р° РїРѕ РґРѕРєСѓРјРµРЅС‚Р°С†С–С— вЂ” Р·СЂРѕР±РёС‚Рё knowledge base РєРµСЂРѕРІР°РЅРѕСЋ РґР»СЏ Р°РіРµРЅС‚С–РІ С–
Р·Р°С„С–РєСЃСѓРІР°С‚Рё Р°СЂС…С–С‚РµРєС‚СѓСЂРЅС– СЂС–С€РµРЅРЅСЏ С‡РµСЂРµР· ADR.

---

## Implemented Capabilities

### offline_topology_preparation

- `status`: implemented
- `commands`:
  - `ingest-topology`
  - `sync-topology`

**Description (ukr):**

РўРѕРїРѕР»РѕРіС–СЏ РїСЂРѕС†РµСЃС–РІ РіРѕС‚СѓС”С‚СЊСЃСЏ offline. Р”Р»СЏ РѕРґРёРЅРёС‡РЅРѕРіРѕ dataset РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ
`ingest-topology`, РґР»СЏ bulk-СЃРёРЅС…СЂРѕРЅС–Р·Р°С†С–С— вЂ” `sync-topology`. Runtime train/eval
РЅРµ РјР°С” РїРµСЂРµР±СѓРґРѕРІСѓРІР°С‚Рё topology Р· raw sources.

### offline_stats_snapshots

- `status`: implemented
- `commands`:
  - `sync-stats`
  - `sync-stats-backfill`
- `snapshot_policy`: immutable_json_asof

**Description (ukr):**

РЎС‚Р°С‚РёСЃС‚РёРєР° РїСЂРѕС†РµСЃС–РІ РіРѕС‚СѓС”С‚СЊСЃСЏ offline С– Р·Р±РµСЂС–РіР°С”С‚СЊСЃСЏ СЏРє immutable snapshots.
`sync-stats-backfill` РІРёРєРѕСЂРёСЃС‚РѕРІСѓС”С‚СЊСЃСЏ РґР»СЏ РїРѕР±СѓРґРѕРІРё С–СЃС‚РѕСЂРёС‡РЅРѕС— timeline
snapshots, РЅРµРѕР±С…С–РґРЅРѕС— РґР»СЏ `strict_asof` РґРѕСЃР»С–РґР¶РµРЅСЊ.

### repository_backed_runtime

- `status`: implemented
- `port`: `IKnowledgeGraphPort`
- `backends`:
  - `in_memory`
  - `file`
  - `neo4j`

**Description (ukr):**

Train/eval runtime СЃРїРѕР¶РёРІР°С” structure С– stats С‡РµСЂРµР· `IKnowledgeGraphPort`.
Backend РјР°С” Р±СѓС‚Рё config-driven: РїРµСЂРµРјРёРєР°РЅРЅСЏ РјС–Р¶ `file`, `in_memory`, `neo4j`
РЅРµ РїРѕРІРёРЅРЅРѕ РІРёРјР°РіР°С‚Рё РїРµСЂРµРїРёСЃСѓРІР°РЅРЅСЏ domain/application Р»РѕРіС–РєРё.

### stats_contract_v1

- `status`: implemented
- `contract_version`: `1.0`
- `producer`: `sync-stats`
- `consumer`: `DynamicGraphBuilder`
- `alignment_profiles`: `legacy_exact`, `safe_normalized`, `research_strict`
- `alignment_service`: `src/domain/services/activity_topology_alignment_service.py`

**Description (ukr):**

Universal stats contract v1 СЂРµР°Р»С–Р·РѕРІР°РЅРѕ: producer-side quality/alignment gates,
`metadata.stats_index`, `metadata.stats_contract`, consumer quality gate С–
config-driven mapping Сѓ `struct_x`.

### gnn_runtime

- `status`: implemented
- `model_families`:
  - `BaselineGATv2`
  - `BaselineGCN`
  - `EOPKGGATv2`
  - `EOPKGGCN`
  - `EOPKGTopologyConditioned`

**Description (ukr):**

РџС–РґС‚СЂРёРјСѓСЋС‚СЊСЃСЏ baseline GNN С– EOPKG-РјРѕРґРµР»С–. EOPKG runtime РјРѕР¶Рµ РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚Рё
`allowed_target_mask`, `structural_edge_index`, `structural_edge_weight`,
`struct_node_to_class_index`, `struct_x` С– snapshot telemetry. Р”Р»СЏ
`EOPKGGATv2` РґРѕСЃС‚СѓРїРЅС– backward-compatible `ClassMeanAttention` /
`ClassMeanConcat` С– РЅРѕРІРёР№ `ClassAwareStructuralScoring`, СЏРєРёР№ РґРѕРґР°С”
node-level structural logits, Р°РіСЂРµРіРѕРІР°РЅС– Сѓ class logits С‡РµСЂРµР·
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
`docs/GNN_LEARNING_METODOLOGY.MD` defines the research roadmap beyond current
fixed-head fusion ablations: topology-conditioned candidate scoring,
semantic-topological candidate prototypes, calibration/audit requirements, and
the staged path toward business-valid zero-shot adaptation.
`training.learning_strategy=standard` preserves current behavior.
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
`training.dynamic_candidate_contract_enabled=true`, train/eval/drift consume
`forward_candidate()` and project topology-local candidate logits back to sparse
fixed-label logits for compatibility with existing CE, mask, calibration, and
drift metrics. Pure candidate-id evaluation without fixed-vocab projection is
still future work. `training.candidate_contract_mode` now separates
`fixed_label`, `fixed_projection`, and reserved `candidate_id` runtime modes.
For `candidate_id`, the DataLoader uses topology-homogeneous batching by
`process_version_idx + stats_snapshot_version_idx` for indexable graph sources,
including in-memory datasets and `ShardedGraphDataset` disk-cache sources. A
safety guard rejects mixed topology batches that still reach the trainer. This
is a validity precondition for future true `[B, C_v]` candidate-level
loss/evaluation. New sharded graph cache entries persist lightweight
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

Train/eval runtime РјР°С” disk-cache РґР»СЏ РїРѕР±СѓРґРѕРІР°РЅРёС… graph datasets С– СЂРµР¶РёРј
sharded disk spill РґР»СЏ РІРµР»РёРєРёС… Р·Р°РїСѓСЃРєС–РІ. РЇРєС‰Рѕ spill СѓРІС–РјРєРЅРµРЅРѕ, runtime РїСЂРёРјСѓСЃРѕРІРѕ
РїРѕС‚СЂРµР±СѓС” cache write mode. `max_ram_gb` С” soft RSS guard: РїСЂРё РїРµСЂРµРІРёС‰РµРЅРЅС– Р»С–РјС–С‚Сѓ
Р±СѓС„РµСЂ РіСЂР°С„С–РІ flushРёС‚СЊСЃСЏ Сѓ shards.

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

Offline preparation С– runtime ML С„С–Р·РёС‡РЅРѕ СЂРѕР·РґС–Р»РµРЅС–. Р¦Рµ РєР»СЋС‡РѕРІРёР№ anti-leakage
С–РЅРІР°СЂС–Р°РЅС‚ MVP2.5.

### immutable_stats_snapshots

- `adr`: `docs/adr/0003-immutable-json-stats-snapshots.md`
- `rule`: snapshots are append-only, JSON-only, no TTL

**Description (ukr):**

Stats snapshots РЅРµ РїРµСЂРµР·Р°РїРёСЃСѓСЋС‚СЊСЃСЏ С– РЅРµ РЅРѕСЂРјР°Р»С–Р·СѓСЋС‚СЊСЃСЏ РІ РѕРєСЂРµРјС– `NodeStat` /
`EdgeStat` graph nodes. РџРѕС‚РѕС‡РЅРёР№ РєРѕРЅС‚СЂР°РєС‚ вЂ” JSON payload + `metadata.stats_index`.

### strict_asof_research_policy

- `adr`: `docs/adr/0004-strict-asof-research-policy.md`
- `recommended_research_config`:
  - `experiment.stats_time_policy: strict_asof`
  - `experiment.on_missing_asof_snapshot: raise`

**Description (ukr):**

Р”Р»СЏ РґРѕРєР°Р·РѕРІРёС… temporal/drift РµРєСЃРїРµСЂРёРјРµРЅС‚С–РІ С‚СЂРµР±Р° РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚Рё `strict_asof`
С– fail-fast РїРѕРІРµРґС–РЅРєСѓ РїСЂРё РІС–РґСЃСѓС‚РЅСЊРѕРјСѓ snapshot. Fallback СЂРµР¶РёРјРё РґРѕРїСѓСЃС‚РёРјС– РґР»СЏ
exploratory runs, Р°Р»Рµ РЅРµ РґР»СЏ С„С–РЅР°Р»СЊРЅРёС… РІРёСЃРЅРѕРІРєС–РІ.

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

РЇРєС‰Рѕ `sync-stats` Р·Р°РїСѓСЃРєР°С”С‚СЊСЃСЏ Р· `--as-of`, snapshot РѕС‚СЂРёРјСѓС” СЏРІРЅРѕ Р·Р°РґР°РЅРёР№ С‡Р°СЃ.
Р‘РµР· `--as-of` timestamp Р±РµСЂРµС‚СЊСЃСЏ РЅРµ СЏРє РїРѕС‚РѕС‡РЅРёР№ UTC, Р° Р· РґР°РЅРёС…: РјР°РєСЃРёРјР°Р»СЊРЅРёР№
`event_ts` РїС–СЃР»СЏ train-cut selection.

### snapshot_lookup_time

- `latest`: load latest structure/snapshot
- `strict_asof`: use prefix last-event timestamp

**Description (ukr):**

РЈ `latest` СЂРµР¶РёРјС– runtime Р±РµСЂРµ РѕСЃС‚Р°РЅРЅС–Р№ РґРѕСЃС‚СѓРїРЅРёР№ snapshot. РЈ `strict_asof`
СЂРµР¶РёРјС– lookup РІРёРєРѕРЅСѓС”С‚СЊСЃСЏ РЅР° timestamp РѕСЃС‚Р°РЅРЅСЊРѕС— РїРѕРґС–С— РїСЂРµС„С–РєСЃСѓ.

### missing_snapshot_policy

- `default_runtime_policy`: `disable_stats`
- `research_grade_policy`: `raise`
- `allowed_values`:
  - `disable_stats`
  - `use_base`
  - `raise`

**Description (ukr):**

РџРѕС‚РѕС‡РЅРёР№ runtime РјРѕР¶Рµ РЅРµ РїР°РґР°С‚Рё РїСЂРё missing/degraded snapshot С– РІРёРјРёРєР°С‚Рё stats
branch. Р”Р»СЏ research-grade temporal runs РїРѕС‚СЂС–Р±РЅРѕ РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚Рё `raise`.

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

MVP1 baseline path РЅРµ РјРѕР¶РЅР° Р»Р°РјР°С‚Рё. Structural/EOPKG РїРѕР»СЏ РјР°СЋС‚СЊ Р±СѓС‚Рё additive:
РІС–РґСЃСѓС‚РЅС–СЃС‚СЊ optional MVP2.5 tensors РЅРµ РїРѕРІРёРЅРЅР° СЂСѓР№РЅСѓРІР°С‚Рё baseline training.

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

РќРѕРІС– config keys РјР°СЋС‚СЊ РјР°С‚Рё С‡С–С‚РєРµ РјС–СЃС†Рµ. РЇРєС‰Рѕ РґРѕРґР°С”С‚СЊСЃСЏ РЅРѕРІРёР№ РєР»СЋС‡ Сѓ
`experiment`, `data`, `mapping`, `model`, `training`, `tracking` Р°Р±Рѕ
`sync_stats`, С‚СЂРµР±Р° РѕРЅРѕРІРёС‚Рё `configs/ui/config_catalog.yaml`.

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

Р¦С– С„Р°РєС‚Рё Р±СѓР»Рё Р·РІС–СЂРµРЅС– Р· РєРѕРґРѕРј РїС–Рґ С‡Р°СЃ РґРѕРєСѓРјРµРЅС‚Р°С†С–Р№РЅРѕРіРѕ audit. Р’РѕРЅРё РЅРµ С” РЅРѕРІРёРјРё
runtime requirements, Р°Р»Рµ РґРѕРїРѕРјР°РіР°СЋС‚СЊ РЅР°СЃС‚СѓРїРЅРёРј Р°РіРµРЅС‚Р°Рј РІС–РґСЂС–Р·РЅРёС‚Рё С„Р°РєС‚РёС‡РЅРёР№
СЃС‚Р°РЅ РІС–Рґ С–СЃС‚РѕСЂРёС‡РЅРёС… worklogs.

---

## Validation Gates

Use project venv:

```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
.\.venv-modern\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv-modern\Scripts\python.exe -m pytest tests/ -v
```

**Description (ukr):**

РЈ Codex sandbox plain `python` Р°Р±Рѕ `py -3` РјРѕР¶СѓС‚СЊ РЅРµ РїСЂР°С†СЋРІР°С‚Рё. РљР°РЅРѕРЅС–С‡РЅРѕ
РІРёРєРѕСЂРёСЃС‚РѕРІСѓРІР°С‚Рё `.\.venv-modern\Scripts\python.exe`. РЇРєС‰Рѕ sandbox Р±Р»РѕРєСѓС” Р·Р°РїСѓСЃРє base
interpreter Р· AppData, С‚СЂРµР±Р° РїРѕРІС‚РѕСЂРёС‚Рё С‚РѕР№ СЃР°РјРёР№ venv command Р· escalation.

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

РџРѕС‚РѕС‡РЅРёР№ С„РѕРєСѓСЃ вЂ” РЅРµ РґРѕРґР°РІР°С‚Рё РЅРѕРІС– РІРµР»РёРєС– runtime С„С–С‡С– Р±РµР· РїРѕС‚СЂРµР±Рё, Р° СЃС‚Р°Р±С–Р»С–Р·СѓРІР°С‚Рё
РґРѕРєСѓРјРµРЅС‚Р°С†С–СЋ, ADR-СЂС–С€РµРЅРЅСЏ С– P0 С‚РµС…Р±РѕСЂРі, СЏРєРёР№ Р±Р»РѕРєСѓС” РґРѕРєР°Р·РѕРІС– РµРєСЃРїРµСЂРёРјРµРЅС‚Рё.

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


