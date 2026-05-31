import os
import torch

checkpoint_dir = "checkpoints"
if os.path.exists(checkpoint_dir):
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".pth"):
            path = os.path.join(checkpoint_dir, f)
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                run_id = checkpoint.get("mlflow_run_id")
                epoch = checkpoint.get("epoch")
                val_loss = checkpoint.get("val_loss")
                print(f"File: {f} | Run ID: {run_id} | Epoch: {epoch} | Val Loss: {val_loss}")
            except Exception as e:
                print(f"File: {f} | Error: {e}")
else:
    print("Checkpoint dir not found")
