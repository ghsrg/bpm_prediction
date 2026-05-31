import torch

def inspect_ckpt_brief(path):
    print(f"\n=== Checkpoint: {path} ===")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        for k in ["epoch", "val_loss", "best_epoch", "best_val_loss", "mlflow_run_id"]:
            if k in checkpoint:
                print(f"  {k}: {checkpoint[k]}")
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state")
        if state_dict:
            print(f"  state_dict keys: {len(state_dict)}")
            # print all key names in state_dict
            print("  state_dict keys list:", list(state_dict.keys()))
    except Exception as e:
        print("Failed to load checkpoint:", e)

inspect_ckpt_brief("checkpoints/_EOPKGTC-UN-42-drift_loan_v1_v4_simulated_GATv2_best.pth")
inspect_ckpt_brief("checkpoints/_EOPKGTC-UN-42_loan_v1_v4_simulated_GATv2_best.pth")
