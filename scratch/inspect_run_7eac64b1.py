import os
from pathlib import Path

def main():
    run_id = "7eac64b18fb141509575a1bf2aa04058"
    run_path = Path("mlruns/854778689611649472") / run_id
    metrics_dir = run_path / "metrics"
    
    print(f"=== Metrics for run {run_id} ===")
    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        return
        
    metrics = sorted(os.listdir(metrics_dir))
    for m in metrics:
        m_file = metrics_dir / m
        content = m_file.read_text().strip().splitlines()
        if content:
            last_line = content[-1].split()
            # MLflow metric format: timestamp value step
            # e.g., 1716912345 0.523 0
            if len(last_line) >= 2:
                val = last_line[1]
                step = last_line[2] if len(last_line) > 2 else "N/A"
                print(f"  {m:50} : {val} (step {step})")

if __name__ == "__main__":
    main()
