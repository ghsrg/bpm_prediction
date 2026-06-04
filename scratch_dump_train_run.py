import mlflow
import os

mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")

run_id = "3ea325191f8442ba9d01d67d9b8085ca"
try:
    run = mlflow.get_run(run_id)
    print("=== RUN INFO ===")
    print(f"Name: {run.info.run_name}")
    print(f"Status: {run.info.status}")
    print("\n=== ALL PARAMS ===")
    for k, v in sorted(run.data.params.items()):
        print(f"  {k}: {v}")
    print("\n=== ALL METRICS ===")
    for k, v in sorted(run.data.metrics.items()):
        print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {e}")
