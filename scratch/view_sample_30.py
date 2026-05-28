import json

trace_file = "outputs/mlflow_trace_fallback/structural_traces_run_5a371502830243a4a9441a872866aee0_pid_27132_rank_0.jsonl"

with open(trace_file, "r") as f:
    lines = list(f)
    print(json.dumps(json.loads(lines[30]), indent=2))
