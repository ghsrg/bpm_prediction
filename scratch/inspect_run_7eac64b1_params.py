from pathlib import Path

def main():
    run_id = "7eac64b18fb141509575a1bf2aa04058"
    run_path = Path("mlruns/854778689611649472") / run_id
    params_dir = run_path / "params"
    
    print(f"=== Parameters for run {run_id} ===")
    if not params_dir.exists():
        print(f"Params directory not found: {params_dir}")
        return
        
    for p in sorted(params_dir.iterdir()):
        print(f"  {p.name:50} : {p.read_text().strip()}")

if __name__ == "__main__":
    main()
