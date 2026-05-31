from pathlib import Path

def main():
    run_id = "7eac64b18fb141509575a1bf2aa04058"
    run_path = Path("mlruns/854778689611649472") / run_id
    metrics_dir = run_path / "metrics"
    
    print("Epoch | Train Loss | Val Loss | Val Macro F1 | Train Macro F1")
    print("-" * 65)
    
    def get_history(metric_name):
        p = metrics_dir / metric_name
        if not p.exists():
            return {}
        lines = p.read_text().strip().splitlines()
        history = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                val = float(parts[1])
                step = int(parts[2])
                history[step] = val
        return history

    train_loss_hist = get_history("train_loss")
    val_loss_hist = get_history("val_loss")
    val_f1_hist = get_history("val_macro_f1")
    train_f1_hist = get_history("train_macro_f1")
    
    all_steps = sorted(list(set(train_loss_hist.keys()) | set(val_loss_hist.keys())))
    for step in all_steps:
        tl = train_loss_hist.get(step, float('nan'))
        vl = val_loss_hist.get(step, float('nan'))
        vf = val_f1_hist.get(step, float('nan'))
        tf = train_f1_hist.get(step, float('nan'))
        print(f"{step:5d} | {tl:10.4f} | {vl:8.4f} | {vf:12.4f} | {tf:14.4f}")

if __name__ == "__main__":
    main()
