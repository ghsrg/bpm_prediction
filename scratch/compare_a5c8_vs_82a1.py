import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

run_a = client.get_run("a5c84b642a6a45bdaa43e769ac48fb58")
run_b = client.get_run("82a1c8e43de64819a64416a4817e5158")

p_a = run_a.data.params
p_b = run_b.data.params

all_keys = sorted(list(set(p_a.keys()) | set(p_b.keys())))
diffs = []
for k in all_keys:
    va = p_a.get(k)
    vb = p_b.get(k)
    if va != vb:
        diffs.append({"param": k, "run_a5c84b64": va, "run_82a1c8e4": vb})

df_diff = pd.DataFrame(diffs)
print(df_diff.to_string())
