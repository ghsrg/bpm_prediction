import json
import os

trace_path = "mlruns/854778689611649472/70b67a6d9e6b4cfea02545f1c158ee0b/artifacts/structural_traces_run_70b67a6d9e6b4cfea02545f1c158ee0b_pid_17620_rank_0.jsonl"

v3_v4_traces = []
with open(trace_path, "r", encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        attrs = t.get("attributes", {})
        version = attrs.get("process_version")
        if version in ("v3", "v4", "v5"):
            v3_v4_traces.append(t)

print(f"Found {len(v3_v4_traces)} traces in v3, v4, v5")

# Save first 5 traces to a file
with open("scratch_v3_v4_traces.json", "w", encoding="utf-8") as out:
    json.dump(v3_v4_traces[:5], out, indent=2)

print("Dumped first 5 to scratch_v3_v4_traces.json")
