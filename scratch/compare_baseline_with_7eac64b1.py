import mlflow
import pandas as pd

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    client = mlflow.tracking.MlflowClient()
    
    run_baseline = client.get_run("48c1c6b4ead94f14af65430c7b19da96")
    run_new = client.get_run("7eac64b18fb141509575a1bf2aa04058")
    
    # Extract params
    p_base = run_baseline.data.params
    p_new = run_new.data.params
    
    # Extract metrics
    m_base = run_baseline.data.metrics
    m_new = run_new.data.metrics
    
    all_keys = sorted(list(set(p_base.keys()) | set(p_new.keys())))
    
    differences = []
    for k in all_keys:
        val_base = p_base.get(k)
        val_new = p_new.get(k)
        if val_base != val_new:
            differences.append({
                "param": k,
                "baseline": val_base,
                "new": val_new
            })
            
    df_diff = pd.DataFrame(differences)
    
    metrics_compare = []
    metric_keys = [
        "best_val_loss",
        "strict_test_accuracy",
        "strict_test_macro_f1",
        "strict_test_f1_v1",
        "strict_test_f1_v2",
        "train_loss",
        "val_loss",
        "test_set_nll",
        "train_candidate_invalid_probability_mass",
        "train_candidate_oos_rate",
        "validation_candidate_invalid_probability_mass",
        "validation_candidate_oos_rate",
        "inference_candidate_invalid_probability_mass",
        "inference_candidate_oos_rate"
    ]
    
    for m in metric_keys:
        metrics_compare.append({
            "metric": m,
            "baseline": m_base.get(m),
            "new": m_new.get(m)
        })
    df_metrics = pd.DataFrame(metrics_compare)
    
    print("=== PARAMETER DIFFERENCES ===")
    print(df_diff.to_string(index=False))
    print("\n=== METRICS COMPARISON ===")
    print(df_metrics.to_string(index=False))

if __name__ == "__main__":
    main()
