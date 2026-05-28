import json

path = r"mlruns\854778689611649472\d75a2bd540fe4fc8b8b4ff71ed7483ca\artifacts\structural_traces_run_d75a2bd540fe4fc8b8b4ff71ed7483ca_pid_10268_rank_0.jsonl"
with open(path, "r", encoding="utf-8") as f:
    for i in range(3):
        line = f.readline()
        if not line:
            break
        data = json.loads(line)
        print(f"=== Line {i} ===")
        # Print keys
        print("Root keys:", list(data.keys()))
        if "inputs" in data:
            print("Inputs keys:", list(data["inputs"].keys()))
            if "sample" in data["inputs"]:
                print("Sample:", data["inputs"]["sample"])
        if "outputs" in data:
            print("Outputs keys:", list(data["outputs"].keys()))
            if "prediction" in data["outputs"]:
                print("Prediction:", data["outputs"]["prediction"])
        if "attributes" in data:
            print("Attributes keys:", list(data["attributes"].keys()))
            print("Attributes:")
            for k, v in data["attributes"].items():
                print(f"  {k}: {v}")
