from pathlib import Path

rd = Path("mlruns/854778689611649472/302fb003f78d468b867a9477e40e9d50/metrics")

def read_metric(name):
    f = rd / name
    if not f.exists():
        return []
    return [float(line.split()[1]) for line in f.read_text(encoding="utf-8").strip().splitlines()]

macro_f1 = read_metric("drift_window_macro_f1")
strict_f1 = read_metric("drift_window_strict_macro_f1")
target_in_mask = read_metric("drift_window_target_in_mask_rate")
pred_in_mask = read_metric("drift_window_pred_in_mask_rate")
ambiguous = read_metric("drift_window_ambiguous_prefix_rate")

print(f"Total windows: {len(macro_f1)}")
print("Macro F1 min, max, first, last:", min(macro_f1), max(macro_f1), macro_f1[0], macro_f1[-1])
print("Strict F1 min, max, first, last:", min(strict_f1), max(strict_f1), strict_f1[0], strict_f1[-1])
print("Target in mask min, max, first, last:", min(target_in_mask), max(target_in_mask), target_in_mask[0], target_in_mask[-1])
print("Pred in mask min, max, first, last:", min(pred_in_mask), max(pred_in_mask), pred_in_mask[0], pred_in_mask[-1])

# Where does macro_f1 drop from 1.0?
drop_idx = [i for i, v in enumerate(macro_f1) if v < 0.999]
print("First window where macro_f1 < 1.0:", drop_idx[0] if drop_idx else "NEVER")
if drop_idx:
    print(f"Window {drop_idx[0]} values: macro_f1={macro_f1[drop_idx[0]]}, strict_f1={strict_f1[drop_idx[0]]}, target_in_mask={target_in_mask[drop_idx[0]]}")
