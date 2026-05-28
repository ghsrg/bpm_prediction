import json
from collections import defaultdict
import numpy as np

path = r"mlruns\854778689611649472\d75a2bd540fe4fc8b8b4ff71ed7483ca\artifacts\structural_traces_run_d75a2bd540fe4fc8b8b4ff71ed7483ca_pid_10268_rank_0.jsonl"

version_diagnostics = defaultdict(list)

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        sample = data.get("inputs", {}).get("sample", {})
        attrs = data.get("attributes", {})
        
        # Determine process version
        version = sample.get("process_version") or attrs.get("process_version") or "__unknown__"
        
        # Extract candidate diagnostics
        # They are in attributes (or in outputs.diagnostics)
        valid_mass = attrs.get("eval_drift_one_pass_candidate_valid_probability_mass")
        if valid_mass is None:
            valid_mass = attrs.get("candidate_valid_probability_mass")
        
        invalid_mass = attrs.get("eval_drift_one_pass_candidate_invalid_probability_mass")
        if invalid_mass is None:
            invalid_mass = attrs.get("candidate_invalid_probability_mass")
            
        score_gap = attrs.get("candidate_score_gap")
        entropy = attrs.get("candidate_prediction_entropy")
        
        # Get target index and prediction index
        prediction = data.get("outputs", {}).get("prediction", {})
        target_label = prediction.get("target_label")
        pred_label = prediction.get("pred_label")
        
        version_diagnostics[version].append({
            "valid_mass": valid_mass,
            "invalid_mass": invalid_mass,
            "score_gap": score_gap,
            "entropy": entropy,
            "target_label": target_label,
            "pred_label": pred_label,
            "pred_in_mask": attrs.get("pred_in_mask", False),
            "target_in_mask": attrs.get("target_in_mask", False),
        })

print("=== Diagnostics by Version from Traces ===")
for version, items in sorted(version_diagnostics.items()):
    total = len(items)
    valid_masses = [x["valid_mass"] for x in items if x["valid_mass"] is not None]
    invalid_masses = [x["invalid_mass"] for x in items if x["invalid_mass"] is not None]
    score_gaps = [x["score_gap"] for x in items if x["score_gap"] is not None]
    entropies = [x["entropy"] for x in items if x["entropy"] is not None]
    
    mean_valid = np.mean(valid_masses) if valid_masses else np.nan
    mean_invalid = np.mean(invalid_masses) if invalid_masses else np.nan
    mean_gap = np.mean(score_gaps) if score_gaps else np.nan
    mean_entropy = np.mean(entropies) if entropies else np.nan
    
    print(f"\nVersion: {version} ({total} samples)")
    print(f"  Mean Valid Prob Mass: {mean_valid:.4f}")
    print(f"  Mean Invalid Prob Mass: {mean_invalid:.4f}")
    print(f"  Mean Score Gap: {mean_gap:.4f}")
    print(f"  Mean Prediction Entropy: {mean_entropy:.4f}")
    
    # Detail on target and predictions
    print("  Samples details (Target -> Pred [In Mask?]):")
    for x in items[:5]:
        print(f"    - {x['target_label']} -> {x['pred_label']} (In Mask: {x['pred_in_mask']}, Target In Mask: {x['target_in_mask']})")
