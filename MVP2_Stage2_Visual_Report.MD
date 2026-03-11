# MVP2 Stage 2 Visual Report

Date: 2026-03-11  
Scope: Research Diagnostics Tool for topology sanity-check (visualization)

## 1. What Was Implemented

### 1.1 TopologyExtractorService visualization capabilities

Updated [topology_extractor_service.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\services\topology_extractor_service.py):

1. Added `fit(train_traces)` alias (delegates to `extract_from_logs`).
2. Added `export_to_networkx(version: str) -> nx.DiGraph`:
   - builds a directed NetworkX graph from `allowed_edges`.
   - raises clear `ValueError` for unknown version.
3. Added `plot_topology(version: str, save_path: str | None = None)`:
   - renders labeled directed graph with arrows via matplotlib.
   - saves PNG when `save_path` is provided.
   - shows interactive window when `save_path` is omitted.

### 1.2 New CLI tool

Created [visualize_topology.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tools\visualize_topology.py):

Arguments:

1. `--config <yaml>` OR `--data <xes>` (mutually exclusive, one required)
2. `--version <kappa>` (default `"1"`)
3. `--out <path.png>` (optional)

Pipeline:

1. Load traces (`XESAdapter`).
2. Build train subset (for `--config`: temporal sort -> macro train split -> fraction).
3. `TopologyExtractorService.fit(train_traces)`.
4. `plot_topology(version, save_path)`.

### 1.3 Main entrypoint integration

Updated [main.py](C:\Users\korsr\PycharmProjects\bpm_prediction\main.py):

1. Added subcommand routing:
   - `python main.py visualize-topology ...`
2. Existing training path remains unchanged:
   - `python main.py --config ...` still uses `src.cli.main()`.

### 1.4 Dependencies

Updated [requirements.txt](C:\Users\korsr\PycharmProjects\bpm_prediction\requirements.txt):

1. Added `networkx>=3.1`
2. `matplotlib` was already present.

## 2. Tests Added/Updated

Updated [test_topology_extractor_service.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\application\test_topology_extractor_service.py):

1. Existing tests kept:
   - unique transitions extraction by version
   - unknown version returns `None`
   - BPMN path raises expected `NotImplementedError`
2. Added tests:
   - `test_export_to_networkx_returns_digraph_with_expected_edges`
   - `test_plot_topology_saves_png`

## 3. Verification Results

### 3.1 Targeted service tests

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/application/test_topology_extractor_service.py -v
```

Result:
- `5 passed`

### 3.2 Regression Shield

Command:

```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:
- `16 passed, 35 deselected`

### 3.3 Full test suite

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:
- `51 passed`

## 4. How To Run the Tool

### 4.1 From experiment config (recommended)

```bash
python main.py visualize-topology --config configs/experiments/01_train_bpi2012.yaml --version 1 --out outputs/topology_v1.png
```

### 4.2 Directly from XES file (no config cascade)

```bash
python main.py visualize-topology --data path/to/log.xes --version 1 --out outputs/topology_v1.png
```

### 4.3 Interactive plot window

```bash
python main.py visualize-topology --config configs/experiments/01_train_bpi2012.yaml --version 1
```

## 5. Can We Run Full MVP2 Validation Now?

Partially:

1. We can already inspect extracted topologies visually and validate extraction sanity.
2. We still need Step 3+ to validate full Zero-Shot injection behavior:
   - DynamicGraphBuilder mask injection
   - EOPKGFusionGNN forward fusion
   - OOS + slicing metrics in evaluator path

