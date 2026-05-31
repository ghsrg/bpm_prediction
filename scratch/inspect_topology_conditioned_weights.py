import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

run_prefixes = {
    "2c866c77 (Succ)": "2c866c77",
    "82a1c8e4 (Fail)": "82a1c8e4",
    "ffce0dbc (Fail)": "ffce0dbc"
}

data = {}
for name, prefix in run_prefixes.items():
    found = False
    for exp in client.search_experiments():
        for r in client.search_runs(exp.experiment_id):
            if r.info.run_id.startswith(prefix):
                data[name] = r.data.params
                found = True
                break
        if found:
            break

all_keys = set()
for p in data.values():
    all_keys.update(p.keys())

rows = []
for k in sorted(list(all_keys)):
    if "topology" in k or "loss" in k or "penalty" in k or "retention" in k or "aux" in k or "xattn" in k:
        row = {"param": k}
        for name in run_prefixes.keys():
            row[name] = data[name].get(k)
        rows.append(row)

df = pd.DataFrame(rows)
print(df.to_string())
