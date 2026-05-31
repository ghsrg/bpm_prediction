import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

run_prefixes = {
    "2c866c77 (Succ)": "2c866c77",
    "48c1c6b4 (Succ)": "48c1c6b4",
    "82a1c8e4 (Fail)": "82a1c8e4",
    "ffce0dbc (Fail)": "ffce0dbc"
}

run_ids = {}
for name, prefix in run_prefixes.items():
    found = False
    for exp in client.search_experiments():
        runs = client.search_runs(exp.experiment_id)
        for r in runs:
            if r.info.run_id.startswith(prefix):
                run_ids[name] = r.info.run_id
                found = True
                break
        if found:
            break
    if not found:
        print(f"Could not find run with prefix {prefix}")

data = {}
for name, rid in run_ids.items():
    run = client.get_run(rid)
    data[name] = run.data.params

all_keys = set()
for p in data.values():
    all_keys.update(p.keys())

diffs = []
for k in sorted(list(all_keys)):
    values = {name: data[name].get(k) for name in run_ids.keys()}
    if len(set(values.values())) > 1:
        row = {"param": k}
        row.update(values)
        diffs.append(row)

df = pd.DataFrame(diffs)
print(df.to_string())
