import torch
from pathlib import Path

p1 = Path("checkpoints/_Base-MN-f1-42_loan_v1_v5_simulated_GATv2_best.pth")
p2 = Path("checkpoints/models/_Base-M-42_loan_v1_v5_best.pth")

d1 = torch.load(str(p1), map_location="cpu", weights_only=False)
d2 = torch.load(str(p2), map_location="cpu", weights_only=False)

print("p1 mlflow_run_id:", d1.get("mlflow_run_id"))
print("p2 mlflow_run_id:", d2.get("mlflow_run_id"))
print("p1 best_val_loss:", d1.get("best_val_loss"))
print("p2 best_val_loss:", d2.get("best_val_loss"))

# Compare state_dict keys and values
s1 = d1["model_state_dict"]
s2 = d2["model_state_dict"]
diff_keys = set(s1.keys()) ^ set(s2.keys())
print("diff keys:", diff_keys)

max_diff = 0.0
for k in s1:
    if k in s2:
        diff = (s1[k] - s2[k]).abs().max().item()
        if diff > max_diff:
            max_diff = diff
print("max weight diff between p1 and p2:", max_diff)
