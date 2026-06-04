import mlflow
import os

mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")

try:
    experiments = mlflow.search_experiments()
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        for idx, row in runs.iterrows():
            run_name = row.get('tags.mlflow.runName', '')
            run_id = row['run_id']
            if 'f02-5-penalty' in str(run_name) and row.get('status', '') == 'FINISHED':
                lr_strat = row.get('params.training.learning_strategy', '')
                print(f"FOUND Training Run {run_id}: Name={run_name}, lr_strat={lr_strat}")
                print("--- Select Metrics ---")
                for k, v in row.items():
                    if any(x in k for x in ['strict_test_macro_f1', 'test_accuracy', 'train_loss', 'val_loss', 'val_macro_f1']):
                        print(f"  {k}: {v}")
except Exception as e:
    print(f"Error: {e}")
