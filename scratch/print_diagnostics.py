import json

path = r"mlruns\854778689611649472\d75a2bd540fe4fc8b8b4ff71ed7483ca\artifacts\structural_traces_run_d75a2bd540fe4fc8b8b4ff71ed7483ca_pid_10268_rank_0.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i in range(2):
        line = f.readline()
        if not line:
            break
        data = json.loads(line)
        print(f"\n=== Diagnostics for Sample {i} ===")
        print(json.dumps(data.get("outputs", {}).get("diagnostics", {}), indent=2))
