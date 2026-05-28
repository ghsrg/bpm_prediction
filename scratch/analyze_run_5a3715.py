import json
from collections import Counter, defaultdict

trace_file = "outputs/mlflow_trace_fallback/structural_traces_run_5a371502830243a4a9441a872866aee0_pid_27132_rank_0.jsonl"

stats_by_version = defaultdict(list)
predictions_by_version = defaultdict(list)

with open(trace_file, "r") as f:
    for line in f:
        data = json.loads(line)
        attrs = data.get("attributes", {})
        version = attrs.get("process_version", "unknown")
        
        # Determine classification correctness
        strict_correct = attrs.get("strict_correct", False)
        # OOS prediction
        outputs = data.get("outputs", {})
        pred_label = outputs.get("prediction", {}).get("pred_label")
        target_label = outputs.get("prediction", {}).get("target_label")
        
        # Candidate diagnostics
        candidate_dynamic_count = attrs.get("candidate_dynamic_count", 0)
        candidate_prediction_entropy = attrs.get("candidate_prediction_entropy", 0.0)
        candidate_score_gap = attrs.get("candidate_score_gap", 0.0)
        
        # Check if prediction is unseen (index -1 in candidate_class_index)
        diag = data.get("outputs", {}).get("diagnostics", {}).get("topology_conditioned_candidate_scoring", {})
        cand_ids = diag.get("candidate_ids", [])
        cand_is_unseen = diag.get("candidate_is_unseen", [])
        
        pred_is_unseen = False
        target_is_unseen = False
        
        if pred_label in cand_ids:
            p_idx = cand_ids.index(pred_label)
            if p_idx < len(cand_is_unseen):
                pred_is_unseen = cand_is_unseen[p_idx]
                
        if target_label in cand_ids:
            t_idx = cand_ids.index(target_label)
            if t_idx < len(cand_is_unseen):
                target_is_unseen = cand_is_unseen[t_idx]
                
        stats_by_version[version].append({
            "strict_correct": strict_correct,
            "pred_is_unseen": pred_is_unseen,
            "target_is_unseen": target_is_unseen,
            "entropy": candidate_prediction_entropy,
            "gap": candidate_score_gap,
            "pred_label": pred_label,
            "target_label": target_label,
            "candidate_dynamic_count": candidate_dynamic_count
        })

print("=== Run 5a371502830243a4a9441a872866aee0 Analysis ===")
for version, samples in sorted(stats_by_version.items()):
    n = len(samples)
    correct = sum(1 for s in samples if s["strict_correct"])
    acc = correct / n if n > 0 else 0.0
    
    pred_unseen_count = sum(1 for s in samples if s["pred_is_unseen"])
    target_unseen_count = sum(1 for s in samples if s["target_is_unseen"])
    
    avg_entropy = sum(s["entropy"] for s in samples) / n if n > 0 else 0.0
    avg_gap = sum(s["gap"] for s in samples) / n if n > 0 else 0.0
    dynamic_count = samples[0]["candidate_dynamic_count"] if n > 0 else 0
    
    print(f"\nVersion: {version} ({n} samples)")
    print(f"  Accuracy (micro-F1): {acc:.4f} ({correct}/{n})")
    print(f"  Candidate count in topology: {dynamic_count}")
    print(f"  Target is unseen activity rate: {target_unseen_count / n:.4%}")
    print(f"  Predicted unseen activity rate: {pred_unseen_count / n:.4%}")
    print(f"  Average candidate prediction entropy: {avg_entropy:.4f}")
    print(f"  Average candidate score gap: {avg_gap:.4f}")
    
    # Print sample of errors or unseen targets
    unseen_targets = [s for s in samples if s["target_is_unseen"]]
    if unseen_targets:
        print("  Sample of unseen target cases:")
        for ut in unseen_targets[:3]:
            print(f"    Target: {ut['target_label']}, Predicted: {ut['pred_label']} (unseen: {ut['pred_is_unseen']})")
            
    errors = [s for s in samples if not s["strict_correct"]]
    if errors:
        print("  Sample of prediction errors:")
        for err in errors[:3]:
            print(f"    Target: {err['target_label']}, Predicted: {err['pred_label']} (unseen: {err['pred_is_unseen']})")
