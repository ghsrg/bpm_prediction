from pathlib import Path
import json

mlruns = Path("mlruns")

def get_run_dir(run_id):
    for exp_dir in mlruns.iterdir():
        if exp_dir.is_dir():
            rd = exp_dir / run_id
            if rd.is_dir():
                return rd
    return None

dir_old = get_run_dir("302fb003f78d468b867a9477e40e9d50")
dir_new = get_run_dir("f71502fd16624ce2b838f50f91dd160a")

params_old = {p.name: p.read_text(encoding="utf-8").strip() for p in (dir_old / "params").iterdir()}
params_new = {p.name: p.read_text(encoding="utf-8").strip() for p in (dir_new / "params").iterdir()}

print(f"Old run params count: {len(params_old)}, New run params count: {len(params_new)}")

# Find differences in params
all_keys = sorted(set(params_old.keys()) | set(params_new.keys()))
diffs = []
for k in all_keys:
    val_old = params_old.get(k, "<MISSING>")
    val_new = params_new.get(k, "<MISSING>")
    if val_old != val_new:
        diffs.append((k, val_old, val_new))

print(f"Total param differences: {len(diffs)}")
for k, vo, vn in diffs:
    print(f"DIFF: {k}\n   OLD: {vo}\n   NEW: {vn}")
