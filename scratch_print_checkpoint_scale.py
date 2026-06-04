import torch
import os

ckpts = {
    "Topology-Conditioned (penalty)": "checkpoints/_EOPKG_TC-UN-f02-5-penalty-42_loan_v1_v5_simulated_GATv2_best.pth",
    "Standard": "checkpoints/_EOPKG_TC-UN-f02tp-5-42_loan_v1_v5_simulated_GATv2_best.pth"
}

for name, checkpoint_path in ckpts.items():
    print(f"\n=== {name} ===")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        continue

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
        
        key = "impulse_scale_raw"
        if key in state_dict:
            val = state_dict[key].item()
            print(f"  impulse_scale_raw: {val:.6f}")
            clamped = max(0.0, min(val, 2.0))
            print(f"  clamped value: {clamped:.6f}")
        else:
            print(f"  Key '{key}' not found.")
            
    except Exception as e:
        print(f"  Error reading checkpoint: {e}")

