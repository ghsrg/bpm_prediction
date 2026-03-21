# MVP2.5 Stage4.2 - Stat Experiment Configs Report

Date: 2026-03-20  
Status: Completed

## Created data configs

1. `configs/data/mvp2_5_stage4_2_camunda_files_stat.yaml`
2. `configs/data/mvp2_5_stage4_2_xes_neo4j_stat.yaml`

Both include:
1. `mapping.graph_feature_mapping.enabled: true`
2. Tier-A node/edge stats mapping for canonical structural stats features
3. `stats_quality_gate.on_fail: allow_with_warning`
4. Neo4j knowledge backend + strict load

## Created experiment configs

1. `configs/experiments/mvp2_5_stage4_2_eopkg_files_stat.yaml`
2. `configs/experiments/mvp2_5_stage4_2_baseline_files_stat.yaml`
3. `configs/experiments/mvp2_5_stage4_2_eopkg_xes_stat.yaml`
4. `configs/experiments/mvp2_5_stage4_2_baseline_xes_stat.yaml`

Common settings:
1. `experiment.stats_time_policy: strict_asof`
2. `training.retrain: true`
3. Proper model include split:
   - EOPKG -> `../model/gatv2_eopkg.yaml`
   - Baseline -> `../model/gatv2_baseline.yaml`

## Validation

Validated all four configs with `src.cli.load_yaml_config(...)`.

Checks passed:
1. include resolution
2. `experiment.name` and `stats_time_policy`
3. `mapping.graph_feature_mapping.enabled=true`
4. `node_numeric` features present
