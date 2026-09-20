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

print("OLD mapping:")
print(yaml.dump(c_old.get("mapping")))

print("\nNEW mapping:")
print(yaml.dump(c_new.get("mapping")))
