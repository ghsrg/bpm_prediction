import os

run_dirs = [
    "mlruns/854778689611649472/2c866c7723594324ae59d970b3288774",
    "mlruns/854778689611649472/82a1c8e43de64819a64416a4817e5158"
]

for run_dir in run_dirs:
    print(f"\n=== Run dir: {run_dir} ===")
    if os.path.exists(run_dir):
        artifacts_dir = os.path.join(run_dir, "artifacts")
        if os.path.exists(artifacts_dir):
            for root, dirs, files in os.walk(artifacts_dir):
                for f in files:
                    path = os.path.join(root, f)
                    rel = os.path.relpath(path, artifacts_dir)
                    print(f"  {rel} (size={os.path.getsize(path)})")
        else:
            print("  No artifacts directory")
    else:
        print("  Not found")
