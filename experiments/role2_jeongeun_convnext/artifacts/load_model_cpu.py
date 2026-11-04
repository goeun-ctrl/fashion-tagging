# Trusted checkpoints only: verify SHA-256 before loading this full training checkpoint.
import torch
import timm
from pathlib import Path

root = Path(__file__).resolve().parent
checkpoint = torch.load(root / "weights" / "best.pt", map_location="cpu", weights_only=False)
# The checkpoint contains model, optimizer, scheduler, scaler and RNG state.
# Model parameters are under the model key, not the whole checkpoint dictionary.
model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k_384", pretrained=False, num_classes=27, global_pool="avg")
model.load_state_dict(checkpoint["model"], strict=True)
model = model.cpu().eval()
print("Loaded on CPU; no inference executed:", sum(p.numel() for p in model.parameters()))
