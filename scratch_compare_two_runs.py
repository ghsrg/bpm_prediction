import mlflow
import os

mlruns_dir = os.path.abspath("mlruns")
if os.path.exists(mlruns_dir):
    mlflow.set_tracking_uri(f"file:///{mlruns_dir}")
    print(f"Tracking URI set to: file:///{mlruns_dir}")

run_1_id = "70b67a6d9e6b4cfea02545f1c158ee0b"
run_2_id = "b0d9aead56764126b784cb8ef1c8ef21"

def print_comparison(id1, id2):
    try:
        run1 = mlflow.get_run(id1)
        run2 = mlflow.get_run(id2)
        
        print("\n=== RUNS GENERAL INFO ===")
        print(f"Run 1 ID: {id1} | Name: {run1.info.run_name} | Status: {run1.info.status}")
        print(f"Run 2 ID: {id2} | Name: {run2.info.run_name} | Status: {run2.info.status}")
        
        print("\n=== PARAMETERS DIFFERENCE ===")
        params_keys = set(run1.data.params.keys()) | set(run2.data.params.keys())
        for k in sorted(params_keys):
            val1 = run1.data.params.get(k, "NOT_SET")
            val2 = run2.data.params.get(k, "NOT_SET")
            if val1 != val2:
                print(f"  {k}:")
                print(f"    Run 1: {val1}")
                print(f"    Run 2: {val2}")
                
        print("\n=== METRICS DIFFERENCE ===")
        metrics_keys = set(run1.data.metrics.keys()) | set(run2.data.metrics.keys())
        for k in sorted(metrics_keys):
            val1 = run1.data.metrics.get(k, "NOT_SET")
            val2 = run2.data.metrics.get(k, "NOT_SET")
            if val1 != val2 or val1 != "NOT_SET":
                val1_str = f"{val1:.6f}" if isinstance(val1, (int, float)) else str(val1)
                val2_str = f"{val2:.6f}" if isinstance(val2, (int, float)) else str(val2)
                # Filter to interesting metrics
                if any(x in k for x in ['f1', 'accuracy', 'oos', 'mask', 'loss', 'ece', 'nll', 'entropy', 'probability', 'margin', 'unseen']):
                    print(f"  {k:50} | Run 1: {val1_str:10} | Run 2: {val2_str:10}")
                    
    except Exception as e:
        print(f"Error fetching/comparing runs: {e}")

print_comparison(run_1_id, run_2_id)
