import mlflow

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

experiments = client.search_experiments()
for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    print(f"\nExperiment: {exp.name}")
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        frac = params.get("experiment.fraction") or params.get("fraction")
        retrain = params.get("experiment.retrain") or params.get("training.retrain") or params.get("retrain")
        mode = params.get("experiment.mode") or params.get("mode")
        model_type = params.get("model.type") or params.get("model_type")
        
        # We are interested in fraction 0.2 runs
        if frac == "0.2":
            print(f"  Run: {run.info.run_id[:8]} | Mode: {mode} | Model: {model_type} | Retrain: {retrain} | Strategy: {params.get('training.learning_strategy')} | Val Loss: {metrics.get('val_loss')} | Val F1: {metrics.get('val_macro_f1')}")
