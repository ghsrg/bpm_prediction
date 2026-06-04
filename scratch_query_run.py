import mlflow
import json
import os

run_id = "70b67a6d9e6b4cfea02545f1c158ee0b"

# Set tracking URI to local mlruns if it exists
mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    print(f"Tracking URI set to: file:///{mlruns_dir}")
else:
    print("mlruns directory not found in current directory")

try:
    run = mlflow.get_run(run_id)
    print("\n=== RUN METADATA ===")
    print(f"Run ID: {run.info.run_id}")
    print(f"Status: {run.info.status}")
    print(f"Start Time: {run.info.start_time}")
    print(f"End Time: {run.info.end_time}")
    
    print("\n=== PARAMETERS ===")
    for k, v in sorted(run.data.params.items()):
        print(f"  {k}: {v}")
        
    print("\n=== METRICS ===")
    for k, v in sorted(run.data.metrics.items()):
        print(f"  {k}: {v}")
        
except Exception as e:
    print(f"Error fetching run {run_id}: {e}")
