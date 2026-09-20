from pathlib import Path

def inspect_run_artifacts_and_metrics(run_id):
    for exp_dir in Path("mlruns").iterdir():
        if not exp_dir.is_dir():
            continue
        rd = exp_dir / run_id
        if rd.is_dir():
            print(f"=== RUN {run_id} ({rd}) ===")
            art_dir = rd / "artifacts"
            print("Artifacts:")
            for a in art_dir.rglob("*"):
                if a.is_file():
                    print(f"  {a.relative_to(art_dir)} ({a.stat().st_size} bytes)")
            
            metrics_dir = rd / "metrics"
            print("Metrics available:")
            for m in sorted(metrics_dir.glob("*")):
                lines = m.read_text(encoding="utf-8").strip().splitlines()
                val0 = lines[0].split()[1] if lines else "N/A"
                print(f"  {m.name}: {len(lines)} steps, first={val0}")

inspect_run_artifacts_and_metrics("302fb003f78d468b867a9477e40e9d50")
print("="*60)
inspect_run_artifacts_and_metrics("f71502fd16624ce2b838f50f91dd160a")
