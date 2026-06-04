import json
import os

trace_path = "mlruns/854778689611649472/70b67a6d9e6b4cfea02545f1c158ee0b/artifacts/structural_traces_run_70b67a6d9e6b4cfea02545f1c158ee0b_pid_17620_rank_0.jsonl"
output_path = "scratch_first_trace.json"

if not os.path.exists(trace_path):
    print("Trace file not found")
    exit()

with open(trace_path, "r", encoding="utf-8") as f:
    first_line = f.readline()
    obj = json.loads(first_line)
    
with open(output_path, "w", encoding="utf-8") as out:
    json.dump(obj, out, indent=2)

print("First trace dumped to scratch_first_trace.json")
