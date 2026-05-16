# MVP2.5 TopologyStateGraphEncoder Fusion Implementation Report

## Summary

Implemented `model.fusion_mode=TopologyStateGraphEncoder` for `EOPKGGATv2`.

This mode is a structural graph baseline:

```text
structural identity + optional struct_x stats + struct_prefix_state_x
  -> structural GNN over structural_edge_index
  -> mean graph pooling
  -> fusion head with zero observed context
  -> classifier
```

It is intended to mirror the old etalon/article-style setup where the process
structure is the graph being encoded before classification. It is not a final
drift-transfer mechanism.

## Files Changed

- `src/domain/models/eopkg_models.py`
- `src/application/use_cases/trainer.py`
- `src/cli.py`
- `configs/ui/config_catalog.yaml`
- `tests/domain/test_eopkg_topology_state_encoder.py`
- `tests/domain/test_eopkg_models_forward.py`
- `tests/domain/test_eopkg_gatv2_dual_encoder.py`
- `tests/application/test_cli_flatten_config.py`
- `tests/application/test_trainer_forward_stats_logging.py`
- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/current/project-state.md`
- `docs/dissertation_changes.MD`

## Runtime Semantics

Canonical config:

```yaml
experiment:
  structural_mode: true

model:
  type: EOPKGGATv2
  fusion_mode: TopologyStateGraphEncoder
  topology_graph_pooling: mean
```

Aliases:

```text
TopologyStateGraphEncoder
topology_state_graph_encoder
TopologyGraphEncoder
topology_graph_encoder
StructuralGraphEncoder
```

The internal canonical value is:

```text
topology_state_graph_encoder
```

The mode requires:

```text
structural_edge_index
struct_prefix_state_x
```

Missing `struct_prefix_state_x` raises:

```text
TopologyStateGraphEncoder requires struct_prefix_state_x.
```

`experiment.structural_mode=false` still routes to observed-only behavior.

## Difference From Other Modes

`TopologyStateEncoder`:

```text
topology nodes -> structural GNN -> node logits -> node-to-class pooling
```

`TopologyStateGraphEncoder`:

```text
topology nodes -> structural GNN -> graph-level context -> classifier
```

`StructuralPriorEncoder`:

```text
observed prefix encoder + mean structural context -> fusion -> classifier
```

`TopologyStateGraphEncoder` is stricter as a structural baseline: the observed
instance graph features do not participate in prediction in this mode.

## Config Keys

Added:

```yaml
model.topology_graph_pooling: mean
```

Updated:

```yaml
model.fusion_mode enum includes TopologyStateGraphEncoder
model.topology_state_beta applies to TopologyStateEncoder and TopologyStateGraphEncoder
model.topology_state_beta_max applies to TopologyStateEncoder and TopologyStateGraphEncoder
model.topology_state_gate_init_bias applies to TopologyStateEncoder and TopologyStateGraphEncoder
model.topology_state_dropout applies to TopologyStateEncoder and TopologyStateGraphEncoder
```

## Diagnostics

Trainer now logs graph-level topology diagnostics:

```text
topology_graph_context_mean_abs
topology_graph_context_max_abs
topology_graph_logits_mean_abs
topology_graph_entropy
```

Existing topology-state prefix/gate diagnostics are reused:

```text
topology_state_prefix_mean_abs
topology_state_prefix_max_abs
topology_state_gate_mean
topology_state_gate_max
```

## Tests Added

Added tests for:

- fusion mode aliases;
- unsupported `model.topology_graph_pooling`;
- graph-level logits and diagnostics;
- fail-fast missing `struct_prefix_state_x`;
- observed-input invariance for this pure structural graph mode;
- `structural_mode=false` fallback to observed-only;
- CLI factory forwarding of `topology_graph_pooling`;
- trainer logging/tracker metrics for graph-level topology diagnostics.

## Verification

RED verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_state_encoder.py tests/domain/test_eopkg_gatv2_dual_encoder.py::test_eopkg_gatv2_rejects_unknown_topology_graph_pooling tests/application/test_cli_flatten_config.py::test_build_model_factory_kwargs_passes_topology_graph_config tests/application/test_trainer_forward_stats_logging.py::test_trainer_logs_topology_graph_diagnostics -v
```

Result before implementation:

```text
12 failed, 8 passed
```

Focused model/CLI/trainer verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_state_encoder.py tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py tests/application/test_cli_flatten_config.py tests/application/test_trainer_forward_stats_logging.py -v
```

Result:

```text
95 passed, 11 warnings
```

Contract/cache verification:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/contracts/test_graph_tensor_contract_mvp2_optional_fields.py tests/domain/test_dynamic_graph_builder_topology_state.py tests/application/test_trainer_topology_state_contract.py tests/application/test_trainer_preloaded_data.py -v
```

Result:

```text
8 passed
```

Config catalog parse:

```powershell
.\.venv\Scripts\python.exe -c "import yaml; yaml.safe_load(open('configs/ui/config_catalog.yaml', encoding='utf-8')); print('config_catalog.yaml OK')"
```

Result:

```text
config_catalog.yaml OK
```

Architecture guard:

```powershell
.\.venv\Scripts\python.exe tools\architecture_guard.py
```

Result:

```text
[ARCH_GUARD] OK: architecture dependency rules satisfied.
```

Full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:

```text
374 passed, 1 failed, 13 warnings
```

The remaining failure is outside this feature and existed as known residual
debt:

```text
tests/domain/test_dynamic_graph_builder_cache_policy.py::test_cache_policy_full_caches_dto_for_strict_asof
assert repo.calls_asof == 1
actual: 5
```

## How To Run First Experiment

Recommended same-version check:

```yaml
experiment:
  structural_mode: true

model:
  type: EOPKGGATv2
  fusion_mode: TopologyStateGraphEncoder
  topology_graph_pooling: mean
  topology_state_beta: 0.5
  topology_state_beta_max: 2.0
  topology_state_gate_init_bias: -2.0
  structural_stats_beta: 0.0
```

Then compare against:

```text
Base-UN
StructuralPriorEncoder/concat
TopologyStateEncoder
ClassAwareStructuralScoring
```

Start with `structural_stats_beta=0.0` to isolate topology identity and prefix
state. Then test `0.1`, `0.3`, and `1.0`.

## Known Limits

- Old graph caches without `struct_prefix_state_x` cannot run this mode.
- Duplicate activity identity ambiguity remains active debt.
- Hard `structural_edge_index` can still overfit to seen topology versions.
- This mode is a same-version structural usefulness baseline, not a transfer
  solution.

## Documentation Drift Check

Updated:

- `configs/ui/config_catalog.yaml`
- `docs/GNN_RUNTIME_MVP2_5.MD`
- `docs/current/project-state.md`
- `docs/dissertation_changes.MD`
- `docs/worklogs/MVP2_5_Topology_State_Graph_Encoder_Fusion_Report_2026-05-16.MD`
