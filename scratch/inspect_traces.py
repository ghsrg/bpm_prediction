import os
import json

run_id = "d75a2bd540fe4fc8b8b4ff71ed7483ca"
# Let's find the artifacts directory for this run
mlruns_dir = "mlruns"
found_path = None
for root, dirs, files in os.walk(mlruns_dir):
    if run_id in root and "artifacts" in root:
        for file in files:
            if file.endswith(".jsonl"):
                found_path = os.path.join(root, file)
                break
        if found_path:
            break

if not found_path:
    print("Could not find the traces jsonl file on disk.")
else:
    print(f"Found traces file at: {found_path}")
    print("\n=== Reading first 5 trace events ===")
    with open(found_path, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            data = json.loads(line)
            # Print a clean summary of the trace
            print(f"\nTrace {count + 1}:")
            print(f"  process_version: {data.get('process_version')}")
            print(f"  case_id: {data.get('case_id')}")
            print(f"  prefix_len: {data.get('prefix_len')}")
            print(f"  target_label: {data.get('target_label')}")
            print(f"  predicted_label: {data.get('predicted_label')}")
            
            # Print candidate info
            candidates = data.get("candidates", [])
            print(f"  Candidates ({len(candidates)}):")
            for cand in candidates[:10]:
                print(f"    - ID: {cand.get('candidate_id')}, Label: {cand.get('candidate_label')}, ClassIdx: {cand.get('candidate_class_idx')}, Logit: {cand.get('logit'):.4f}, Prob: {cand.get('probability'):.4f}, Target: {cand.get('is_target')}, Allowed: {cand.get('is_allowed')}")
            if len(candidates) > 10:
                print(f"    ... and {len(candidates) - 10} more")
            
            count += 1
            if count >= 5:
                break
