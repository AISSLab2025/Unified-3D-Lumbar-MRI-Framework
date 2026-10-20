"""Check required files and imports without loading checkpoints or patient data."""
import importlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
failed = []
manifest = json.loads((root / "checkpoints/manifest.json").read_text())
for name in manifest["files"]:
    if not (root / name).is_file():
        failed.append("Missing weight: " + name)
for name in ["torch", "torchvision", "cv2", "pydicom", "fastapi", "multipart",
             "timm", "mamba_ssm", "MultiScaleDeformableAttention"]:
    try:
        module = importlib.import_module(name)
        if name == "torch" and not module.cuda.is_available():
            failed.append("CUDA is unavailable to PyTorch")
    except Exception as exc:
        failed.append(f"Import {name}: {exc}")
print("\n".join(failed) if failed else "Setup checks passed. Next: run a study through /predict.")
raise SystemExit(bool(failed))
