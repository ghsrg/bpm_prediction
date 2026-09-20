import yaml
from pathlib import Path

def load_yaml(run_id, filename):
    for exp_dir in Path("mlruns").iterdir():
        if exp_dir.is_dir():
            rd = exp_dir / run_id
            if rd.is_dir():
                f = rd / "artifacts" / filename
                if f.exists():
                    return yaml.safe_load(f.read_text(encoding="utf-8"))
    return None

c_old = load_yaml("302fb003f78d468b867a9477e40e9d50", "ui_run_fqg84z0j.yaml")
c_new = load_yaml("f71502fd16624ce2b838f50f91dd160a", "ui_run_7ufaxtv6.yaml")

def diff_dicts(d1, d2, path=""):
    diffs = []
    keys = sorted(set(d1.keys()) | set(d2.keys()))
    for k in keys:
        p = f"{path}.{k}" if path else k
        v1 = d1.get(k)
        v2 = d2.get(k)
        if isinstance(v1, dict) and isinstance(v2, dict):
            diffs.extend(diff_dicts(v1, v2, p))
        elif v1 != v2:
            diffs.append((p, v1, v2))
    return diffs

diffs = diff_dicts(c_old, c_new)
print(f"Total YAML config diffs: {len(diffs)}")
for p, v1, v2 in diffs:
    print(f"KEY: {p}\n   OLD: {v1}\n   NEW: {v2}")
