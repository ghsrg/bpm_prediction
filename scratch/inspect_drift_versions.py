import json
import os
from collections import defaultdict
from sklearn.metrics import f1_score, accuracy_score

path = r"mlruns\854778689611649472\d75a2bd540fe4fc8b8b4ff71ed7483ca\artifacts\structural_traces_run_d75a2bd540fe4fc8b8b4ff71ed7483ca_pid_10268_rank_0.jsonl"

version_data = defaultdict(list)

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        sample = data.get("inputs", {}).get("sample", {})
        prediction = data.get("outputs", {}).get("prediction", {})
        attrs = data.get("attributes", {})
        
        # Get target and prediction labels/indices
        target_idx = prediction.get("target_index")
        pred_idx = prediction.get("pred_index")
        
        # Determine process version
        version = sample.get("process_version") or attrs.get("process_version") or "__unknown__"
        
        if target_idx is not None and pred_idx is not None:
            version_data[version].append({
                "target_idx": target_idx,
                "pred_idx": pred_idx,
                "strict_correct": prediction.get("strict_correct", False),
                "confidence": prediction.get("confidence", 0.0),
                "pred_in_mask": attrs.get("pred_in_mask", attrs.get("prediction_in_mask", False)),
                "target_in_mask": attrs.get("target_in_mask", False),
                "strict_error_but_allowed": attrs.get("strict_error_but_allowed", False),
            })

print("=== Prediction Summary by Version ===")
for version, items in sorted(version_data.items()):
    total = len(items)
    targets = [x["target_idx"] for x in items]
    preds = [x["pred_idx"] for x in items]
    
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    
    target_in_mask_rate = sum(1 for x in items if x["target_in_mask"]) / total if total else 0
    pred_in_mask_rate = sum(1 for x in items if x["pred_in_mask"]) / total if total else 0
    strict_error_but_allowed_rate = sum(1 for x in items if x["strict_error_but_allowed"]) / total if total else 0
    oos_rate = 1.0 - pred_in_mask_rate
    
    print(f"\nVersion: {version} (Total samples: {total})")
    print(f"  Strict Accuracy: {acc:.4f}")
    print(f"  Strict Macro F1: {macro_f1:.4f}")
    print(f"  Target In Mask Rate: {target_in_mask_rate:.4f}")
    print(f"  Pred In Mask Rate: {pred_in_mask_rate:.4f}")
    print(f"  OOS prediction rate: {oos_rate:.4f}")
    print(f"  Strict Error but Allowed Rate: {strict_error_but_allowed_rate:.4f}")
