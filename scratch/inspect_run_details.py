import mlflow
import os
import json

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

run_id = "d75a2bd540fe4fc8b8b4ff71ed7483ca"
try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"Error fetching run {run_id}: {e}")
    # Let's list all runs in all experiments to see if the ID is slightly different or check what runs exist.
    print("Listing all available run IDs:")
    for exp in client.search_experiments():
        for r in client.search_runs(exp.experiment_id):
            print(f"Experiment {exp.name} ({exp.experiment_id}): Run {r.info.run_id} ({r.info.run_name})")
    sys.exit(1)

print("=== Run Info ===")
print(f"Run ID: {run.info.run_id}")
print(f"Run Name: {run.info.run_name}")
print(f"Status: {run.info.status}")
print(f"Start Time: {run.info.start_time}")

print("\n=== Parameters ===")
for k, v in sorted(run.data.params.items()):
    print(f"  {k}: {v}")

print("\n=== Metrics ===")
for k, v in sorted(run.data.metrics.items()):
    print(f"  {k}: {v}")

print("\n=== Logged Artifacts ===")
artifacts = client.list_artifacts(run_id)
for art in artifacts:
    print(f"  {art.path} (is_dir: {art.is_dir}, size: {art.file_size if hasattr(art, 'file_size') else 'unknown'})")
    if not art.is_dir and art.path.endswith(".json") or art.path.endswith(".jsonl") or art.path.endswith(".txt"):
        # We can download it later if needed
        pass
