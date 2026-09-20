from pathlib import Path

mlruns = Path("mlruns")
runs = []
for exp_dir in mlruns.iterdir():
    if not exp_dir.is_dir() or exp_dir.name.startswith("."):
        continue
    for run_dir in exp_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        meta_file = run_dir / "meta.yaml"
        if meta_file.exists():
            runs.append((meta_file.stat().st_mtime, exp_dir.name, run_dir.name, run_dir))

runs.sort(key=lambda x: x[0], reverse=True)
print(f"Total runs found: {len(runs)}")
print("\nTop 15 most recent runs:")
for mtime, exp_id, run_id, run_dir in runs[:15]:
    # read run_name from meta or tags
    run_name = ""
    tags_name = run_dir / "tags" / "mlflow.runName"
    if tags_name.exists():
        run_name = tags_name.read_text(encoding="utf-8").strip()
    print(f"  {run_id} (exp={exp_id}, name={run_name})")

# Check if f71502fd16624ce2b838f50f91dd160a exists anywhere in runs
matches = [r for r in runs if "f71502" in r[2]]
print("\nMatches for f71502:", matches)
if matches:
    target_dir = matches[0][3]
    params_dir = target_dir / "params"
    metrics_dir = target_dir / "metrics"
    print("\n--- Params for target run ---")
    for p in sorted(params_dir.iterdir()):
        print(f"  {p.name} = {p.read_text(encoding='utf-8').strip()}")
    print("\n--- Drift macro f1 metrics ---")
    f1_file = metrics_dir / "drift_window_macro_f1"
    if f1_file.exists():
        lines = f1_file.read_text(encoding="utf-8").strip().splitlines()
        print(f"Count: {len(lines)}")
        print("First 10:", [l.split()[1] for l in lines[:10]])
        print("Last 10:", [l.split()[1] for l in lines[-10:]])
