import torch
import math

path_a = "checkpoints/_EOPKGTC-UN-42_loan_v1_v4_simulated_GATv2_best.pth"
path_b = "checkpoints/_EOPKGTC-UN-42-drift_loan_v1_v4_simulated_GATv2_best.pth"

ckpt_a = torch.load(path_a, map_location="cpu", weights_only=False)
ckpt_b = torch.load(path_b, map_location="cpu", weights_only=False)

state_a = ckpt_a["model_state_dict"]
state_b = ckpt_b["model_state_dict"]

# Print epochs and validation losses
print(f"Ckpt A: Epoch={ckpt_a.get('epoch')}, Val Loss={ckpt_a.get('val_loss')}")
print(f"Ckpt B: Epoch={ckpt_b.get('epoch')}, Val Loss={ckpt_b.get('val_loss')}")

# Compare some key weights
print("\n=== Weight Comparison ===")
all_keys = sorted(list(set(state_a.keys()) & set(state_b.keys())))
for k in all_keys[:10]:
    w_a = state_a[k]
    w_b = state_b[k]
    if isinstance(w_a, torch.Tensor) and isinstance(w_b, torch.Tensor):
        diff = torch.abs(w_a - w_b).mean().item()
        print(f"Key: {k} | Shape: {w_a.shape} | Mean Abs Diff: {diff:.6f}")
