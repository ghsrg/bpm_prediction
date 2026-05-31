import os
import json
import yaml
from pathlib import Path

mlruns_dir = Path("mlruns")
runs = []

for exp_dir in mlruns_dir.iterdir():
    if not exp_dir.is_dir() or exp_dir.name in (".trash", "models", "0"):
        continue
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "tags":
            continue
        meta_yaml = run_dir / "meta.yaml"
        if meta_yaml.exists():
            with open(meta_yaml, "r") as f:
                meta = yaml.safe_load(f)
            
            # get name
            tags_dir = run_dir / "tags"
            run_name = ""
            if tags_dir.exists():
                name_file = tags_dir / "mlflow.runName"
                if name_file.exists():
                    run_name = name_file.read_text().strip()
            
            runs.append({
                "run_id": meta.get("run_id"),
                "experiment_id": meta.get("experiment_id"),
                "name": run_name,
                "start_time": meta.get("start_time"),
                "status": meta.get("status"),
                "path": str(run_dir)
            })

# sort by start time desc
runs.sort(key=lambda x: x["start_time"] or 0, reverse=True)

print("Recent 10 runs:")
for r in runs[:10]:
    import datetime
    start_dt = datetime.datetime.fromtimestamp(r["start_time"]/1000) if r["start_time"] else "N/A"
    print(f"ID: {r['run_id']} | Name: {r['name']} | Started: {start_dt} | Status: {r['status']}")
