import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

run_a = client.get_run("2c866c7723594324ae59d970b3288774")
run_b = client.get_run("82a1c8e43de64819a64416a4817e5158")

p_a = run_a.data.params
p_b = run_b.data.params

all_keys = sorted(list(set(p_a.keys()) | set(p_b.keys())))
diffs = []
for k in all_keys:
    va = p_a.get(k)
    vb = p_b.get(k)
    if va != vb:
        diffs.append({"param": k, "run_2c866c77": va, "run_82a1c8e4": vb})

df_diff = pd.DataFrame(diffs)
print("=== PARAM DIFFERENCES ===")
print(df_diff.to_string())

print("\n=== TAGS ===")
for run_name, run in [("2c866c77", run_a), ("82a1c8e4", run_b)]:
    print(f"\nRun {run_name} tags:")
    for k, v in sorted(run.data.tags.items()):
        if not k.startswith("mlflow.sys"):
            print(f"  {k}: {v}")
