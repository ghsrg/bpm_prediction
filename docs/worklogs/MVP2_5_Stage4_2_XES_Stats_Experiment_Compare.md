# MVP2.5 Stage 4.2 - XES Stats Experiment Compare

Date: 2026-03-20  
Status: Completed

## Goal

Investigate why `edge_stats_json`, `gnn_features_json`, `metadata_json` looked mostly zero for `BPI_Challenge_2012` and verify with controlled comparison.

## Setup

Process:
1. `process_name = BPI_Challenge_2012`
2. `version_key = BPI_Challenge_2012`
3. `as_of = 2012-03-14T15:04:54.681000+00:00`
4. Data volume from run:
   - `total_traces = 13087`
   - `total_events = 164506`
   - `history_coverage_percent = 100.0`

## Experiment A (before alignment)

Config state:
1. `mapping.xes_adapter.use_classifier = true`
2. Activities from XES became form `"<concept:name>+COMPLETE"`.

Observed overlap:
1. `unique XES activities = 23`
2. `activity ∩ ProcessNode.node_id = 0`
3. `activity ∩ ProcessNode.name = 0`

Snapshot metrics (`k000004`):
1. `edge_stats_json`: `0 / 3312` non-zero (`0.0000`)
2. `gnn_features_json`: `8 / 1112` non-zero (`0.0072`)
3. `metadata_json`: `395 / 10337` non-zero (`0.0382`)

## Experiment B (alignment enabled)

Config change:
1. `mapping.xes_adapter.use_classifier = false`
2. Activities become raw `concept:name`, matching graph node ids for this topology.

Observed overlap:
1. `unique XES activities = 23`
2. `activity ∩ ProcessNode.node_id = 23` (`1.0`)
3. `activity ∩ ProcessNode.name = 23` (`1.0`)

Snapshot metrics (`k000006`, same as_of):
1. `edge_stats_json`: `2766 / 3312` non-zero (`0.8351`)
2. `gnn_features_json`: `1008 / 1112` non-zero (`0.9065`)
3. `metadata_json`: `8173 / 10337` non-zero (`0.7907`)

## Control sanity check

Synthetic aligned mini-case (`A -> B`) was executed with exact id match:
1. `edge_stats` non-zero ratio: `0.75`
2. `gnn_features` non-zero ratio: `0.75`
3. `metadata` non-zero ratio: `0.5625`

This confirms the pipeline computes non-zero stats correctly when mapping is aligned.

## Conclusion

Root cause of near-zero stats was not missing events, but **activity-to-node mismatch** caused by classifier-generated activity ids (`+COMPLETE` suffix).

For this dataset/process, keeping `use_classifier=false` is correct for sync-stats against BPMN topology in Neo4j.

## Additional note on Neo4j warning spam

During experiment, DBMS warnings for missing optional properties were observed.
A query fix was applied in repository loading path to use dynamic property access for optional keys; current full run completed without warning spam.
