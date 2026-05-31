import os
from pathlib import Path

run_ids = ["5b421c1a35e7415aaf80593220efeae8", "0a485575e8374637b59a4fb800d1005a"]
mlruns_dir = Path("mlruns")

for rid in run_ids:
    print(f"=== Run {rid} ===")
    run_dir = None
    for exp_dir in mlruns_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name in (".trash", "models", "0"):
            continue
        candidate = exp_dir / rid
        if candidate.exists():
            run_dir = candidate
            break
    
    if not run_dir:
        print("Run dir not found")
        continue
    
    # Check params
    params_dir = run_dir / "params"
    if params_dir.exists():
        params = os.listdir(params_dir)
        print("Number of params:", len(params))
        # Print a few key params
        key_params = ["experiment.fraction", "training.device", "model.type", "experiment.mode", "training.epochs"]
        for kp in key_params:
            kp_path = params_dir / kp
            if kp_path.exists():
                print(f"  {kp}: {kp_path.read_text().strip()}")
    
    # Check metrics
    metrics_dir = run_dir / "metrics"
    if metrics_dir.exists():
        metrics = os.listdir(metrics_dir)
        print("Metrics:", metrics)
        # Print the last value of each metric
        for m in metrics:
            m_path = metrics_dir / m
            if m_path.exists():
                lines = m_path.read_text().splitlines()
                if lines:
                    print(f"  {m} (last line): {lines[-1]}")
    
    # Check artifacts
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        print("Artifacts structure:")
        for root, dirs, files in os.walk(artifacts_dir):
            for f in files:
                rel_path = os.path.relpath(os.path.join(root, f), artifacts_dir)
                print(f"  - {rel_path} ({os.path.getsize(os.path.join(root, f))} bytes)")
