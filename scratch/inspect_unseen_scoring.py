import json

trace_file = "outputs/mlflow_trace_fallback/structural_traces_run_5a371502830243a4a9441a872866aee0_pid_27132_rank_0.jsonl"

with open(trace_file, "r") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        attrs = data.get("attributes", {})
        version = attrs.get("process_version")
        
        diag = data.get("outputs", {}).get("diagnostics", {}).get("topology_conditioned_candidate_scoring", {})
        cand_labels = diag.get("candidate_labels", [])
        cand_is_unseen = diag.get("candidate_is_unseen", [])
        
        outputs = data.get("outputs", {})
        prediction = outputs.get("prediction", {})
        pred_label = prediction.get("pred_label")
        target_label = prediction.get("target_label")
        confidence = prediction.get("confidence")
        
        cand_pred_score = diag.get("candidate_pred_score")
        cand_target_score = diag.get("candidate_target_score")
        
        # Let's inspect samples of v3 or v4, or any sample with a highly negative target score
        if version in ("v3", "v4") or (cand_target_score is not None and cand_target_score < -100):
            print(f"Sample {i} (Version: {version}):")
            print(f"  Target: {target_label} (score: {cand_target_score})")
            print(f"  Predicted: {pred_label} (score: {cand_pred_score}, confidence: {confidence:.4f})")
            
            # Print top 5 candidates by score/logit if available in outputs
            top_k = prediction.get("top_k", [])
            print("  Top 5 Predicted:")
            for tk in top_k[:5]:
                print(f"    Label: {tk['label']}, Prob: {tk['probability']:.4f}")
                
            # If target_label is in cand_labels, check if it was allowed by mask
            if target_label in cand_labels:
                t_idx = cand_labels.index(target_label)
                # Check target mask in inputs
                has_cand_mask = data.get("inputs", {}).get("contract", {}).get("has_candidate_allowed_target_mask", False)
                print(f"  Target in candidate list: True (index: {t_idx}, unseen: {cand_is_unseen[t_idx] if t_idx < len(cand_is_unseen) else 'N/A'})")
            else:
                print(f"  Target in candidate list: FALSE")
            print("-" * 50)
