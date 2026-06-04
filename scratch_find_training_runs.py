import mlflow
import os

mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    print(f"Tracking URI set to: file:///{mlruns_dir}")

# We want to find runs that contain EOPKG_TC and f02 and trained on loan_v1_v5_simulated
try:
    # Get all experiments
    experiments = mlflow.search_experiments()
    print(f"Found {len(experiments)} experiments")
    
    for exp in experiments:
        print(f"Experiment ID: {exp.experiment_id}, Name: {exp.name}")
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(f"  Total runs in this experiment: {len(runs)}")
        
        # Look for matching runs
        for idx, row in runs.iterrows():
            run_id = row['run_id']
            run_name = row.get('tags.mlflow.runName', '')
            experiment_name = row.get('params.experiment.name', '')
            model_type = row.get('params.model.type', '')
            
            if '70b67a6d' in run_id or '70b67a6d9e6b4cfea02545f1c158ee0b' == run_id:
                print(f"    -> FOUND target run {run_id}: Name={run_name}, ExpName={experiment_name}")
            
            # Let's search for training runs of EOPKG_TC-UN-f02
            if 'EOPKG_TC-UN-f02' in str(run_name) or 'EOPKG_TC-UN-f02' in str(experiment_name):
                status = row.get('status', '')
                lr_strat = row.get('params.training.learning_strategy', '')
                print(f"    Run {run_id} ({status}): Name={run_name}, lr_strat={lr_strat}")
                
except Exception as e:
    print(f"Error querying runs: {e}")
