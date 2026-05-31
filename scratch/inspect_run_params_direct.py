import mlflow

mlflow.set_tracking_uri("file:./mlruns")
client = mlflow.tracking.MlflowClient()

for rid in ["2c866c7723594324ae59d970b3288774", "82a1c8e43de64819a64416a4817e5158"]:
    run = client.get_run(rid)
    print(f"\n=== Run {rid[:8]} ===")
    for k in ["experiment.retrain", "training.retrain", "experiment.load_checkpoint", "resume_checkpoint_path", "model.hidden_dim", "fraction", "experiment.fraction_strategy"]:
        print(f"  {k}: {run.data.params.get(k)}")
