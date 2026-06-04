import mlflow
import os

mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")

run_id = "b8daa11d5ffa4a3db6d6804a4a1bc345"

try:
    run = mlflow.get_run(run_id)
    print("=== RUN INFO ===")
    print(f"Name: {run.info.run_name}")
    print(f"Status: {run.info.status}")
    print("\n=== SELECTED PARAMS ===")
    for k in sorted(run.data.params.keys()):
        val = run.data.params[k]
        if any(x in k for x in ['experiment', 'training', 'model']):
            print(f"  {k}: {val}")
            
    print("\n=== METRICS ===")
    for k, v in sorted(run.data.metrics.items()):
        if any(x in k for x in ['f1', 'accuracy', 'oos', 'mask', 'loss']):
            print(f"  {k}: {v}")
            
except Exception as e:
    print(f"Error: {e}")
