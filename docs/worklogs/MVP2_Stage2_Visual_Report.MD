# MVP2 Stage 2 Visual Report

Date: 2026-03-12  
Scope: Research Diagnostics Tool for topology sanity-check (visualization)

## 1. What Was Implemented

### 1.1 TopologyExtractorService visualization capabilities

Updated [topology_extractor_service.py](C:\Users\korsr\PycharmProjects\bpm_prediction\src\application\services\topology_extractor_service.py):

1. `plot_topology()` migrated from NetworkX/Matplotlib to **PM4Py DFG visualizer**.
2. Added internal DFG adapter `_build_dfg_payload(version)`:
   - builds `Dict[Tuple[str, str], int]` edge frequencies from extracted transition counts.
   - provides `start_activities` and `end_activities` maps for PM4Py node coloring.
3. `extract_from_logs()` now stores real transition frequencies in `ProcessStructureDTO.edge_statistics`.
4. Added noise filtering in `plot_topology(version, save_path, min_edge_freq=1)`:
   - transitions with frequency `< min_edge_freq` are removed before rendering.
   - if filtering removes all transitions, method raises clear `ValueError`.
5. Added explicit runtime error for missing Graphviz executable (`dot`) to make setup failures actionable.
6. `extract_from_bpmn()` now has an architecture note: BPMN visualization path will use `pm4py.visualization.bpmn.visualizer`.

### 1.2 New CLI tool

Created [visualize_topology.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tools\visualize_topology.py):

Arguments:

1. `--config <yaml>` OR `--data <xes>` (mutually exclusive, one required)
2. `--version <kappa>` (default `"1"`)
3. `--out <path.png>` (optional)
4. `--min-freq <int>` (default `1`) for rare-edge filtering.

Pipeline:

1. Load traces (`XESAdapter`).
2. Build train subset (for `--config`: temporal sort -> macro train split -> fraction).
3. `TopologyExtractorService.fit(train_traces)`.
4. `plot_topology(version, save_path, min_edge_freq)`.

### 1.3 Main entrypoint integration

Updated [main.py](C:\Users\korsr\PycharmProjects\bpm_prediction\main.py):

1. Added subcommand routing:
   - `python main.py visualize-topology ...`
2. Existing training path remains unchanged:
   - `python main.py --config ...` still uses `src.cli.main()`.

### 1.4 Dependencies

Updated [requirements.txt](C:\Users\korsr\PycharmProjects\bpm_prediction\requirements.txt):

1. Replaced `networkx` with `pm4py>=2.7.0`.
2. `matplotlib` remains in requirements because it is still used in project utilities/tests.
3. Important runtime prerequisite: install **Graphviz** system package and ensure `dot` is available in `PATH`.

## 2. Tests Added/Updated

Updated [test_topology_extractor_service.py](C:\Users\korsr\PycharmProjects\bpm_prediction\tests\application\test_topology_extractor_service.py):

1. Existing tests kept:
   - unique transitions extraction by version
   - unknown version returns `None`
   - BPMN path raises expected `NotImplementedError`
2. Added tests:
   - `test_build_dfg_payload_returns_expected_edges_and_start_end_activity_maps`
   - `test_plot_topology_saves_png_with_pm4py_visualizer` (mocked PM4Py visualizer API)
   - `test_plot_topology_raises_when_frequency_filter_removes_all_edges`

## 3. Verification Results

### 3.1 Targeted service tests

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/application/test_topology_extractor_service.py -v
```

Result:
- `7 passed`

### 3.2 Regression Shield

Command:

```bash
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```

Result:
- `16 passed, 39 deselected`

### 3.3 Full test suite

Command:

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Result:
- `56 passed`

### 3.4 CLI visualization run (PM4Py path)

Command:

```bash
python main.py visualize-topology --config configs/experiments/01_train_bpi2012.yaml --version 1 --out outputs/topology_v1_pm4py.png
```

Observed behavior:

1. Smart fallback selected available version (`BPI_Challenge_2012`) when requested `1` was absent.
2. PM4Py DFG rendering was invoked.
3. Rendering failed on this machine due to missing Graphviz binary:
   - `RuntimeError: Graphviz executable 'dot' was not found. Install Graphviz and add it to PATH to render PM4Py DFG visuals.`

Conclusion:

- Code path is correct and integrated.
- Local environment needs Graphviz installation for actual PNG/PDF generation.

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

### 4.4 Frequency-filtered clean DFG (recommended for dissertation figures)

```bash
python main.py visualize-topology --config configs/experiments/01_train_bpi2012.yaml --version 1 --min-freq 50 --out outputs/topology_v1_min50.png
```

If you get `Graphviz executable 'dot' was not found`, install Graphviz and restart terminal/IDE.

Windows install examples:

```bash
winget install Graphviz.Graphviz
```

or

```bash
choco install graphviz -y
```

## 5. Can We Run Full MVP2 Validation Now?

Partially:

1. We can already inspect extracted topologies visually and validate extraction sanity.
2. We still need Step 3+ to validate full Zero-Shot injection behavior:
   - DynamicGraphBuilder mask injection
   - EOPKGFusionGNN forward fusion
   - OOS + slicing metrics in evaluator path
