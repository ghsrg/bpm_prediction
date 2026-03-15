# EVF_MVP2_5.MD

Updated: 2026-03-15  
Status: ACTIVE

## 1. Purpose
Evaluation framework rules for MVP2.5 after decoupling topology ingestion from model training.

## 2. Scientific validity goals
- eliminate data leakage from future traces,
- keep reproducible topology artifacts,
- ensure fair baseline vs structure-enhanced comparison.

## 3. Leakage prevention policy

### 3.1 Topology ingestion split
Default ingestion scope is `train` subset only.

Implication:
- structure artifacts are built without eval/test traces,
- evaluation windows cannot leak future transition knowledge.

### 3.2 Temporal split discipline
Train/eval pipeline still applies chronological split independently from ingestion phase.

Rule:
- no mixing of eval traces into train subset,
- no random split for drift studies.

## 4. Fair model comparison protocol
For each experiment pair (baseline vs EOPKG):
1. same dataset and same experiment split config,
2. same seed,
3. same optimizer and epochs,
4. only structural augmentation differs.

## 5. Metrics
Mandatory:
- Accuracy
- F1 macro
- ECE
- OOS (only if mask is available)

Optional sliced metrics:
- by process version,
- by prefix length bins.

## 6. Artifact traceability
Each run should log:
- full config file,
- knowledge backend and path,
- topology versions available,
- git commit and seed.

## 7. Runtime diagnostics quality gates
For Camunda runtime data:
- monitor `history_coverage_percent`,
- monitor `cleaned_instances_percent`,
- track `fallback_triggered`.

If fallback ratio is high for target dataset, report must label results as degraded-confidence.

## 8. Reproducibility checklist
Before publishing experiment result:
1. run `ingest-topology` with frozen config,
2. confirm artifact exists in `data/knowledge_graph/...`,
3. run training with same config and seed,
4. run regression shield tests.

## 9. Required validation commands
```bash
python main.py ingest-topology --config <exp.yaml>
python main.py --config <exp.yaml>
.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v
```
