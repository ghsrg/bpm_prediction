import mlflow

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

run_id = "48c1c6b4ead94f14af65430c7b19da96"
try:
    run = client.get_run(run_id)
    print("=== Parameters ===")
    for k, v in sorted(run.data.params.items()):
        if "candidate" in k or "topology" in k or "strategy" in k or "checkpoint" in k:
            print(f"  {k}: {v}")
    print("=== Metrics ===")
    for k, v in sorted(run.data.metrics.items()):
        print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {e}")
