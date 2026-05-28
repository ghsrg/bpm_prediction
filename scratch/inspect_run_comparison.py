import mlflow
import os

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

train_id = "2c866c7723594324ae59d970b3288774"
drift_id = "d75a2bd540fe4fc8b8b4ff71ed7483ca"

def print_run_details(run_id, title):
    print(f"\n======================================")
    print(f"=== {title} ({run_id}) ===")
    print(f"======================================")
    try:
        run = client.get_run(run_id)
        print("--- Status ---")
        print(f"  Status: {run.info.status}")
        
        print("\n--- Key Parameters ---")
        param_keys = [
            "experiment.mode",
            "experiment.load_checkpoint",
            "model.type",
            "model.fusion_mode",
            "training.learning_strategy",
            "training.candidate_contract_mode",
            "training.candidate_identity_mode",
            "training.candidate_batch_topology_policy",
            "experiment.statistic_enabled",
            "experiment.structural_mode",
            "experiment.stats_time_policy",
        ]
        for pk in param_keys:
            if pk in run.data.params:
                print(f"  {pk}: {run.data.params[pk]}")
            else:
                print(f"  {pk}: <NOT SET>")
                
        print("\n--- Key Metrics ---")
        metric_keys = [
            "test_macro_f1",
            "strict_test_macro_f1",
            "mean_drift_strict_f1",
            "drift_window_strict_macro_f1",
            "drift_window_target_in_mask_rate",
            "drift_window_test_oos",
        ]
        for mk in metric_keys:
            # Check history first
            try:
                hist = client.get_metric_history(run_id, mk)
                if hist:
                    values = [h.value for h in hist]
                    print(f"  {mk}: last={values[-1]:.6f}, min={min(values):.6f}, max={max(values):.6f}, count={len(values)}")
                else:
                    if mk in run.data.metrics:
                        print(f"  {mk}: {run.data.metrics[mk]:.6f} (no history)")
            except Exception:
                if mk in run.data.metrics:
                    print(f"  {mk}: {run.data.metrics[mk]:.6f}")
    except Exception as e:
        print(f"Error reading run {run_id}: {e}")

print_run_details(train_id, "TRAINING RUN")
print_run_details(drift_id, "DRIFT EVALUATION RUN")
