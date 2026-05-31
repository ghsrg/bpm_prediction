import yaml

path_2c = "mlruns/854778689611649472/2c866c7723594324ae59d970b3288774/artifacts/ui_run_jpv_4utw.yaml"
path_82 = "mlruns/854778689611649472/82a1c8e43de64819a64416a4817e5158/artifacts/ui_run_xgln0rzu.yaml"

with open(path_2c, "r", encoding="utf-8") as f:
    cfg_2c = yaml.safe_load(f)

with open(path_82, "r", encoding="utf-8") as f:
    cfg_82 = yaml.safe_load(f)

def dict_diff(d1, d2, path=""):
    diffs = []
    for k in sorted(list(set(d1.keys()) | set(d2.keys()))):
        if k not in d1:
            diffs.append((f"{path}.{k}" if path else k, None, d2[k]))
        elif k not in d2:
            diffs.append((f"{path}.{k}" if path else k, d1[k], None))
        elif isinstance(d1[k], dict) and isinstance(d2[k], dict):
            diffs.extend(dict_diff(d1[k], d2[k], f"{path}.{k}" if path else k))
        elif d1[k] != d2[k]:
            diffs.append((f"{path}.{k}" if path else k, d1[k], d2[k]))
    return diffs

diffs = dict_diff(cfg_2c, cfg_82)
print("=== YAML Differences ===")
for p, v1, v2 in diffs:
    print(f"{p}: 2c866c77={v1} vs 82a1c8e4={v2}")
