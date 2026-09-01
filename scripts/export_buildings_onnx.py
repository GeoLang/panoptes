import json
import sys
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

# usage: export_buildings_onnx.py [checkout of giswqs/whu-building-unetplusplus-efficientnet-b4] [output.onnx]
MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("whu-model")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("panoptes-buildings-v1.onnx")
TILE = 512

config = json.loads((MODEL_DIR / "config.json").read_text())
assert config["architecture"] == "unetplusplus"
assert config["encoder_name"] == "efficientnet-b4"

model = smp.UnetPlusPlus(
    encoder_name=config["encoder_name"],
    encoder_weights=None,
    in_channels=config["num_channels"],
    classes=config["num_classes"],
)
state = torch.load(MODEL_DIR / "model.pth", map_location="cpu", weights_only=True)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state)
model.eval()

example = torch.zeros(1, 3, TILE, TILE)
with torch.no_grad():
    out = model(example)
print("torch output shape:", tuple(out.shape))

torch.onnx.export(
    model,
    example,
    str(OUT),
    input_names=["image"],
    output_names=["logits"],
    opset_version=17,
    dynamo=False,
)
print("wrote", OUT, OUT.stat().st_size, "bytes")
