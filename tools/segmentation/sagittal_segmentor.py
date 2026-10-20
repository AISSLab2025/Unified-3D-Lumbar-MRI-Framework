import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.segmentation.aghm_runtime.model import build_model as build_aghm_model


_MODEL_CACHE = {}
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_AGHM_CHECKPOINT_PATH = (
    _PROJECT_ROOT
    / "checkpoints"
    / "segmentation"
    / "Sagital_AGMH"
    / "best.pt"
)

DEFAULT_AGHM_CONFIG_PATH = (
    _PROJECT_ROOT
    / "checkpoints"
    / "segmentation"
    / "Sagital_AGMH"
    / "inference_config.json"
)

DEFAULT_AGHM_CONFIG = {
    "model": {
        "base_dim": 12,
        "curve": "hilbert",
        "d_conv": 2,
        "d_state": 8,
        "depth": 3,
        "expand": 2,
        "in_channels": 1,
        "out_channels": 5,
        "return_aux": False,
        "sequence_kernel": 7,
        "use_anatomy_gating": True,
        "use_backward_mamba": True,
        "use_forward_mamba": True,
        "variant": "binary_aghm",
    },
    "threshold": 0.5,
}

CLASS_NAMES = {
    0: "Background",
    1: "Posterior_A",
    2: "Posterior_B",
    3: "VB",
    4: "IVD",
    5: "Sacrum",
}

CLASS_COLORS = np.array([
    [0, 0, 0],
    [51, 51, 255],
    [51, 255, 51],
    [255, 153, 51],
    [51, 255, 255],
    [255, 51, 255],
], dtype=np.uint8)


def normalize_path(path_str):
    if not path_str:
        return Path()

    path_str = str(path_str)

    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        path_str = "C:/" + path_str[7:]

    elif os.name != "nt" and (
        path_str.startswith("C:/") or path_str.startswith("C:\\")
    ):
        path_str = "/mnt/c/" + path_str[3:].replace("\\", "/")

    return Path(path_str)


def _clean_state_dict_keys(state_dict):
    if not any(str(key).startswith("module.") for key in state_dict.keys()):
        return state_dict

    return {
        str(key).removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_aghm_config(config_path=DEFAULT_AGHM_CONFIG_PATH):
    config_path = normalize_path(config_path)

    if not config_path.exists():
        return DEFAULT_AGHM_CONFIG

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_aghm_model(checkpoint_path, device, model_config=None):
    checkpoint_path = normalize_path(checkpoint_path)
    model_config = model_config or load_aghm_config()
    model_cfg = dict(model_config.get("model", model_config))

    cache_key = f"{checkpoint_path}:{device}:{model_cfg}"

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_aghm_model(model_cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = _clean_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def rotate_for_model(image_or_mask):
    """
    This is the orientation used before model inference:
    rotate 90 clockwise + horizontal flip.
    Same style as your second testing code.
    """
    image_or_mask = cv2.rotate(image_or_mask, cv2.ROTATE_90_CLOCKWISE)
    image_or_mask = cv2.flip(image_or_mask, 1)
    return image_or_mask


def inverse_rotate_from_model(mask):
    """
    Return prediction back to the original DICOM image space.
    Inverse of:
        rotate 90 clockwise + horizontal flip
    """
    mask = cv2.flip(mask, 1)
    mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return mask


def sagittal_segmentation(
    dicom_slice,
    checkpoint_path=DEFAULT_AGHM_CHECKPOINT_PATH,
    output_dir="output/segmentation/sagittal",
    save_overlay=True,
    rotate_input_for_model=False,
    return_info=True,
    model_config=None,
    threshold=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = normalize_path(checkpoint_path)
    output_dir = normalize_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(dicom_slice, "pixel_array"):
        image = dicom_slice.pixel_array
        instance_number = getattr(dicom_slice, "InstanceNumber", "unknown")
    else:
        image = dicom_slice
        instance_number = "array"

    image = image.astype(np.float32)

    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[-1] == 1:
            image = image[:, :, 0]

    orig_h, orig_w = image.shape[:2]

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val > 0:
        image_norm = (image - min_val) / (max_val - min_val)
    else:
        image_norm = np.zeros_like(image, dtype=np.float32)

    # Keep this image normal for final overlay
    final_image = (image_norm * 255).astype(np.uint8)

    # AGMH was trained/evaluated on the upright sagittal PNG orientation.
    # Keep rotation optional for experiments, but do not use the old SegResNet
    # rotation path by default.
    if rotate_input_for_model:
        model_input_image = rotate_for_model(image_norm)
    else:
        model_input_image = image_norm

    model_h, model_w = model_input_image.shape[:2]

    image_resized = cv2.resize(
        model_input_image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    original_image_path = output_dir / "mid_sagittal_original.png"
    cv2.imwrite(str(original_image_path), final_image)

    image_tensor = (
        torch.from_numpy(image_resized)
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    model_config = model_config or load_aghm_config()
    if threshold is None:
        threshold = float(model_config.get("threshold", 0.5))

    model = get_aghm_model(
        checkpoint_path=checkpoint_path,
        device=device,
        model_config=model_config,
    )

    with torch.no_grad():
        logits = model(image_tensor)
        prob_224 = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)

    max_prob_224 = np.max(prob_224, axis=0)
    argmax_class_224 = np.argmax(prob_224, axis=0).astype(np.uint8) + 1
    pred_mask_224 = np.where(
        max_prob_224 >= float(threshold),
        argmax_class_224,
        0,
    ).astype(np.uint8)

    # Resize prediction to rotated model-input size
    pred_mask_model_space = cv2.resize(
        pred_mask_224.astype(np.uint8),
        (model_w, model_h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Bring prediction back to original DICOM image orientation
    if rotate_input_for_model:
        final_mask = inverse_rotate_from_model(pred_mask_model_space)
    else:
        final_mask = pred_mask_model_space

    # Safety resize to original image size
    if final_mask.shape != final_image.shape:
        final_mask = cv2.resize(
            final_mask.astype(np.uint8),
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay_path = None

    if save_overlay:
        image_bgr = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)

        color_mask = CLASS_COLORS[np.clip(final_mask, 0, 5)]
        overlay = image_bgr.copy()

        mask_non_bg = final_mask > 0
        blended = cv2.addWeighted(image_bgr, 0.25, color_mask, 0.75, 0)
        overlay[mask_non_bg] = blended[mask_non_bg]

        cv2.putText(
            overlay,
            " ",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        overlay_path = (
            output_dir
            / f"mid_sagittal_AGMH_overlay_instance_{instance_number}.png"
        )

        cv2.imwrite(str(overlay_path), overlay)

    result = {
        "mask": final_mask,
        "mask_224": pred_mask_224,
        "prob_224": prob_224,
        "mask_model_space": pred_mask_model_space,
        "image": final_image,
        "class_names": CLASS_NAMES,
        "unique_classes": np.unique(final_mask).tolist(),
        "input_shape": (orig_h, orig_w),
        "model_input_shape": image_resized.shape,
        "output_shape": final_mask.shape,
        "rotate_input_for_model": rotate_input_for_model,
        "output_path": str(overlay_path) if overlay_path else None,
        "overlay_path": str(overlay_path) if overlay_path else None,
        "original_image_path": str(original_image_path),
        "checkpoint_path": str(checkpoint_path),
        "model_name": "AGMH",
        "model_runtime": str(Path(__file__).resolve().parent / "aghm_runtime"),
        "config_path": str(DEFAULT_AGHM_CONFIG_PATH),
        "threshold": float(threshold),
    }

    if return_info:
        return result

    return final_mask
