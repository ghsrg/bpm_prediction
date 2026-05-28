import json

path = r"mlruns\854778689611649472\d75a2bd540fe4fc8b8b4ff71ed7483ca\artifacts\structural_traces_run_d75a2bd540fe4fc8b8b4ff71ed7483ca_pid_10268_rank_0.jsonl"
with open(path, "r", encoding="utf-8") as f:
    line = f.readline()
    if line:
        data = json.loads(line)
        print("=== Keys in inputs.contract ===")
        contract = data.get("inputs", {}).get("contract", {})
        for k, v in contract.items():
            if isinstance(v, list):
                print(f"  {k}: list of length {len(v)}")
            else:
                print(f"  {k}: {v}")
