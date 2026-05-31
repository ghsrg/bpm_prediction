import mlflow

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

# find run starting with 48c1c6b4
rid = None
for exp in client.search_experiments():
    for run in client.search_runs(exp.experiment_id):
        if run.info.run_id.startswith("48c1c6b4"):
            rid = run.info.run_id
            break
    if rid:
        break

print(f"Run ID: {rid}")
if rid:
    for metric_name in ["train_loss", "val_loss", "val_macro_f1"]:
        try:
            history = client.get_metric_history(rid, metric_name)
            print(f"\n=== Metric: {metric_name} ===")
            for point in history[:5]:
                print(f"step: {point.step}, value: {point.value}, ts: {point.timestamp}")
        except Exception as e:
            print(f"Error loading metric {metric_name}: {e}")
