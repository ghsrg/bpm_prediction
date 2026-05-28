import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

for run_id in ["82a1c8e43de64819a64416a4817e5158", "265484126b884c918071f618a2d347dc"]:
    try:
        run = client.get_run(run_id)
        print(f"\n==========================================")
        print(f"Run ID: {run.info.run_id}")
        print(f"Name: {run.data.tags.get('mlflow.runName')}")
        print(f"Status: {run.info.status}")
        print(f"Experiment ID: {run.info.experiment_id}")
        
        print("\n--- Key Parameters ---")
        param_keys = [
            "experiment.mode", "experiment.train_ratio", "experiment.load_checkpoint",
            "model.type", "model.fusion_mode", "training.learning_strategy",
            "training.candidate_contract_mode", "training.candidate_identity_mode",
            "training.topology_flow_penalty_weight", "training.topology_conditioning_allowed_set_loss_weight"
        ]
        for k in param_keys:
            if k in run.data.params:
                print(f"  {k}: {run.data.params[k]}")
                
        print("\n--- Key Metrics ---")
        metric_prefixes = [
            "train_", "val_", "drift_window_", "eval_drift_one_pass_"
        ]
        for k, v in sorted(run.data.metrics.items()):
            if any(k.startswith(p) for p in metric_prefixes) or "loss" in k or "f1" in k.lower() or "accuracy" in k.lower():
                print(f"  {k}: {v}")
                
    except Exception as e:
        print(f"Error querying MLflow run {run_id}: {e}")
