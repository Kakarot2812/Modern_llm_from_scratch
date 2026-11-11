import os
import torch

os.makedirs("deploy_weights", exist_ok=True)

# 1. Export SFT model
sft_ckpt = torch.load("checkpoints/sft_best.pt", map_location="cpu", weights_only=False)
torch.save(sft_ckpt["model_state_dict"], "deploy_weights/sft_model.pt")
print(f"Exported SFT weights: {os.path.getsize('deploy_weights/sft_model.pt') / 1e6:.1f} MB")

# 2. Export Base model (optional, for comparison)
base_ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
torch.save(base_ckpt["model_state_dict"], "deploy_weights/base_model.pt")
print(f"Exported Base weights: {os.path.getsize('deploy_weights/base_model.pt') / 1e6:.1f} MB")
