import mlflow
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

experiments = client.search_experiments()
rows = []
for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        
        drift_strict_f1s = []
        try:
            history = client.get_metric_history(run.info.run_id, "drift_window_strict_macro_f1")
            drift_strict_f1s = [m.value for m in history]
        except Exception:
            pass
        mean_drift_strict = sum(drift_strict_f1s) / len(drift_strict_f1s) if drift_strict_f1s else None
        
        rows.append({
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "model_type": params.get("model.type", params.get("model_type")),
            "learning_strategy": params.get("training.learning_strategy"),
            "stats_time_policy": params.get("experiment.stats_time_policy"),
            "candidate_contract_mode": params.get("training.candidate_contract_mode"),
            "candidate_identity_mode": params.get("training.candidate_identity_mode"),
            "batch_topology_policy": params.get("training.candidate_batch_topology_policy"),
            "statistic_enabled": params.get("experiment.statistic_enabled"),
            "load_checkpoint": params.get("experiment.load_checkpoint"),
            "mean_drift_strict_f1": mean_drift_strict,
            "num_drift_windows": len(drift_strict_f1s),
        })

df = pd.DataFrame(rows)
df_sorted = df.dropna(subset=["mean_drift_strict_f1"]).sort_values(by="mean_drift_strict_f1", ascending=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
print(df_sorted.head(40))
