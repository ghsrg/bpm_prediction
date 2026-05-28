import mlflow
from mlflow.tracking import MlflowClient

run_id = "5a371502830243a4a9441a872866aee0"
client = MlflowClient()

try:
    run = client.get_run(run_id)
    print(f"Run ID: {run.info.run_id}")
    print(f"Status: {run.info.status}")
    print(f"Experiment ID: {run.info.experiment_id}")
    
    print("\n--- Parameters ---")
    for k, v in sorted(run.data.params.items()):
        print(f"  {k}: {v}")
        
    print("\n--- Metrics ---")
    for k, v in sorted(run.data.metrics.items()):
        print(f"  {k}: {v}")
        
except Exception as e:
    print(f"Error querying MLflow run: {e}")
