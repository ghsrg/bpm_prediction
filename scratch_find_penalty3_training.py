import mlflow
import os

mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")

try:
    experiments = mlflow.search_experiments()
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        for idx, row in runs.iterrows():
            run_name = row.get('tags.mlflow.runName', '')
            run_id = row['run_id']
            if 'penalty3' in str(run_name) and not 'drift' in str(run_name):
                print(f"\nFOUND Training Run {run_id}: Name={run_name}, Status={row.get('status', '')}")
                print("--- Parameters ---")
                for k in sorted(row.keys()):
                    if k.startswith('params.'):
                        print(f"  {k[7:]}: {row[k]}")
                print("--- Metrics ---")
                for k in sorted(row.keys()):
                    if k.startswith('metrics.') and any(x in k for x in ['f1', 'loss', 'accuracy']):
                        print(f"  {k[8:]}: {row[k]}")
except Exception as e:
    print(f"Error: {e}")
