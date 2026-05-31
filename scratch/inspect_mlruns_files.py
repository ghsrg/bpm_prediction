import os

run_dir = "mlruns/0/2c866c7723594324ae59d970b3288774"
if not os.path.exists(run_dir):
    # Search in other experiment folders
    for exp in os.listdir("mlruns"):
        test_dir = os.path.join("mlruns", exp, "2c866c7723594324ae59d970b3288774")
        if os.path.exists(test_dir):
            run_dir = test_dir
            break

print(f"Run directory: {run_dir}")
if os.path.exists(run_dir):
    print("Files:")
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, run_dir)
            print(f"  {rel} (size={os.path.getsize(path)})")
            
    # Read the config if it was logged as an artifact
    config_path = os.path.join(run_dir, "artifacts", "config.yaml")
    if os.path.exists(config_path):
        print("\n=== config.yaml ===")
        with open(config_path, "r", encoding="utf-8") as f:
            print(f.read()[:500])
else:
    print("Run directory not found")
