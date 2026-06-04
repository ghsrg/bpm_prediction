import json
import os

trace_path = "mlruns/854778689611649472/70b67a6d9e6b4cfea02545f1c158ee0b/artifacts/structural_traces_run_70b67a6d9e6b4cfea02545f1c158ee0b_pid_17620_rank_0.jsonl"

if not os.path.exists(trace_path):
    print("Trace file not found at", trace_path)
    exit()

traces = []
with open(trace_path, "r", encoding="utf-8") as f:
    for line in f:
        traces.append(json.loads(line))

print(f"Loaded {len(traces)} traces")

# Let's inspect the keys of the first trace
if traces:
    print("Keys in trace:", sorted(traces[0].keys()))
    
# Let's count how many had target_in_mask = False
not_in_mask_count = 0
incorrect_pred_count = 0
unseen_target_count = 0
unseen_target_correct = 0

for t in traces:
    attrs = t.get("attributes", {})
    outputs = t.get("outputs", {})
    pred_data = outputs.get("prediction", {})
    
    target_in_mask = attrs.get("target_in_mask", True)
    pred_in_mask = attrs.get("pred_in_mask", True)
    correct = attrs.get("strict_correct", False)
    # Check if target is unseen - wait, does EOPKGTopologyConditioned contract provide it? Let's check candidate_is_unseen
    # We can check if candidate_is_unseen has True for the target_index or check target_is_unseen if it is written in attributes
    target_is_unseen = attrs.get("target_is_unseen", False)
    if target_is_unseen is None:
        target_is_unseen = False
    
    if not target_in_mask:
        not_in_mask_count += 1
    if not correct:
        incorrect_pred_count += 1
    if target_is_unseen:
        unseen_target_count += 1
        if correct:
            unseen_target_correct += 1

print(f"Not in mask count: {not_in_mask_count} / {len(traces)} ({not_in_mask_count/len(traces)*100:.2f}%)")
print(f"Incorrect predictions: {incorrect_pred_count} / {len(traces)} ({incorrect_pred_count/len(traces)*100:.2f}%)")
print(f"Unseen targets: {unseen_target_count} / {len(traces)}")
if unseen_target_count > 0:
    print(f"Unseen targets correct: {unseen_target_correct} / {unseen_target_count} ({unseen_target_correct/unseen_target_count*100:.2f}%)")

# Let's print details of the first 5 traces where target was not in mask
print("\n=== SAMPLE TRACES (TARGET NOT IN MASK) ===")
count = 0
for t in traces:
    attrs = t.get("attributes", {})
    outputs = t.get("outputs", {})
    pred_data = outputs.get("prediction", {})
    target_in_mask = attrs.get("target_in_mask", True)
    
    if not target_in_mask and count < 5:
        print(f"Trace {attrs.get('trace_idx')}, Step {attrs.get('prefix_len')}:")
        print(f"  Target: {pred_data.get('target_label')} (unseen: {attrs.get('target_is_unseen')})")
        print(f"  Predicted: {pred_data.get('pred_label')}")
        print(f"  Current version: {attrs.get('process_version')}")
        print(f"  Allowed candidates: {pred_data.get('top_k_candidate_labels')}")
        count += 1

# Let's print details of the first 5 incorrect predictions
print("\n=== SAMPLE TRACES (INCORRECT PREDICTIONS) ===")
count = 0
for t in traces:
    attrs = t.get("attributes", {})
    outputs = t.get("outputs", {})
    pred_data = outputs.get("prediction", {})
    correct = attrs.get("strict_correct", False)
    
    if not correct and count < 5:
        print(f"Trace {attrs.get('trace_idx')}, Step {attrs.get('prefix_len')}:")
        print(f"  Target: {pred_data.get('target_label')} (unseen: {attrs.get('target_is_unseen')}, in_mask: {attrs.get('target_in_mask')})")
        print(f"  Predicted: {pred_data.get('pred_label')} (in_mask: {attrs.get('prediction_in_mask')})")
        print(f"  Current version: {attrs.get('process_version')}")
        print(f"  Allowed candidates: {pred_data.get('top_k_candidate_labels')}")
        count += 1

