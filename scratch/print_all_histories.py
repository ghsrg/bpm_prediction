import mlflow

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

run_prefixes = {
    "2c866c77 (Succ)": "2c866c77",
    "82a1c8e4 (Fail)": "82a1c8e4",
    "ffce0dbc (Fail)": "ffce0dbc",
    "0f36f16d (Succ)": "0f36f16d",
    "0d2fd387 (Succ)": "0d2fd387"
}

for name, prefix in run_prefixes.items():
    found = False
    for exp in client.search_experiments():
        for r in client.search_runs(exp.experiment_id):
            if r.info.run_id.startswith(prefix):
                rid = r.info.run_id
                found = True
                break
        if found:
            break
    
    if found:
        print(f"\n=== Run: {name} ({rid[:8]}) ===")
        for metric_name in ["train_loss", "val_loss", "val_macro_f1"]:
            try:
                hist = client.get_metric_history(rid, metric_name)
                vals = [round(p.value, 4) for p in hist]
                print(f"  {metric_name} (len={len(vals)}): {vals[:5]} ... {vals[-5:] if len(vals) > 5 else ''}")
            except Exception as e:
                print(f"  Error loading {metric_name}: {e}")
    else:
        print(f"Run {prefix} not found")
