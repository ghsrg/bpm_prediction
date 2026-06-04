import mlflow
import os

mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")

train_run_id = "3ea325191f8442ba9d01d67d9b8085ca"
drift_run_id = "70b67a6d9e6b4cfea02545f1c158ee0b"

def inspect_run(run_id, name):
    try:
        run = mlflow.get_run(run_id)
        print(f"\n==================== {name} ({run_id}) ====================")
        print("--- PARAMS ---")
        for k in ['experiment.mode', 'training.learning_strategy', 'experiment.fraction_strategy', 
                  'experiment.split_strategy', 'experiment.version_scope_policy', 'experiment.train_ratio',
                  'training.topology_flow_penalty_enabled', 'training.topology_flow_penalty_weight',
                  'training.topology_conditioning_allowed_set_loss_enabled', 'training.topology_conditioning_allowed_set_loss_weight',
                  'training.topology_conditioning_wrong_version_negative_enabled', 'training.topology_conditioning_drop_edges_negative_enabled',
                  'model.type', 'model.fusion_mode', 'model.topology_conditioning_mode']:
            val = run.data.params.get(k, 'NOT_SET')
            print(f"  {k}: {val}")
        print("--- SELECT METRICS ---")
        for k, v in sorted(run.data.metrics.items()):
            if 'f1' in k or 'accuracy' in k or 'oos' in k or 'mask' in k or 'loss' in k or 'val' in k or 'test' in k:
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error inspecting run {run_id}: {e}")

inspect_run(train_run_id, "TRAINING RUN")
inspect_run(drift_run_id, "DRIFT EVAL RUN")
