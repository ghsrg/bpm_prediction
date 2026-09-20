import torch
from pathlib import Path

p1 = Path("checkpoints/_Base-MN-f1-42_loan_v1_v5_simulated_GATv2_best.pth")
p2 = Path("checkpoints/models/_Base-M-42_loan_v1_v5_best.pth")

print(f"p1 exists: {p1.exists()}, size: {p1.stat().st_size if p1.exists() else 0}")
print(f"p2 exists: {p2.exists()}, size: {p2.stat().st_size if p2.exists() else 0}")

if p1.exists():
    d1 = torch.load(str(p1), map_location="cpu", weights_only=False)
    print("p1 keys:", list(d1.keys()))
    print("p1 epoch:", d1.get("epoch"))
    print("p1 best_metric:", d1.get("best_metric"))
    print("p1 meta:", d1.get("meta", {}))

if p2.exists():
    d2 = torch.load(str(p2), map_location="cpu", weights_only=False)
    print("p2 keys:", list(d2.keys()))
    print("p2 epoch:", d2.get("epoch"))
    print("p2 best_metric:", d2.get("best_metric"))
    print("p2 meta:", d2.get("meta", {}))
