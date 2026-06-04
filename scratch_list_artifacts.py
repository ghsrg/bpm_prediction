import mlflow
import os

mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")

run_id = "70b67a6d9e6b4cfea02545f1c158ee0b"
client = mlflow.tracking.MlflowClient()

try:
    artifacts = client.list_artifacts(run_id)
    print("=== Artifacts for run ===")
    for art in artifacts:
        print(f"  Path: {art.path}, is_dir: {art.is_dir}, size: {art.file_size}")
        if art.is_dir:
            # list sub-artifacts
            sub_arts = client.list_artifacts(run_id, path=art.path)
            for sub in sub_arts:
                print(f"    Sub-Path: {sub.path}, size: {sub.file_size}")
except Exception as e:
    print(f"Error listing artifacts: {e}")
