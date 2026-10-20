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
    / "Axial_AGMH"
    / "best.pt"
)

DEFAULT_AGHM_CONFIG_PATH = (
    _PROJECT_ROOT
    / "checkpoints"
    / "segmentation"
    / "Axial_AGMH"
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
        "out_channels": 4,
        "return_aux": False,
        "sequence_kernel": 7,
        "use_anatomy_gating": True,
        "use_backward_mamba": True,
        "use_forward_mamba": True,
        "variant": "binary_aghm",
    },
    "threshold": 0.5,
    "class_names": {
        "0": "Background",
        "1": "PE",
        "2": "TS",
        "3": "AAP",
        "4": "IVD",
    },
}

CLASS_COLOR_HEX = {
    0: "#000000",
    1: "#00FF00",
    2: "#00FFFF",
    3: "#FF00FF",
    4: "#FFFF00",
}

SEGMENTATION_IMAGE_ALPHA = 0.25
SEGMENTATION_MASK_ALPHA = 0.75


def _hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return rgb[::-1]


CLASS_COLORS = np.array([
    _hex_to_bgr(CLASS_COLOR_HEX[idx])
    for idx in range(len(CLASS_COLOR_HEX))
], dtype=np.uint8)

DEFAULT_LEVEL_NAMES = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]


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
    image_or_mask = cv2.rotate(image_or_mask, cv2.ROTATE_90_CLOCKWISE)
    image_or_mask = cv2.flip(image_or_mask, 1)
    return image_or_mask


def inverse_rotate_from_model(mask):
    mask = cv2.flip(mask, 1)
    mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return mask


def _dicom_or_array_to_gray(dicom_or_array):
    if hasattr(dicom_or_array, "pixel_array"):
        image = dicom_or_array.pixel_array
        instance_number = getattr(dicom_or_array, "InstanceNumber", "unknown")
    else:
        image = dicom_or_array
        instance_number = "array"

    image = image.astype(np.float32)

    if image.ndim == 3:
        if image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))

        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[-1] == 1:
            image = image[:, :, 0]

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val > 0:
        image_norm = (image - min_val) / (max_val - min_val)
    else:
        image_norm = np.zeros_like(image, dtype=np.float32)

    return image_norm.astype(np.float32), instance_number


def _resize_channels_nearest(mask_channels, size):
    width, height = size
    resized = [
        cv2.resize(
            channel.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        for channel in mask_channels
    ]
    return np.stack(resized, axis=0).astype(np.uint8)


def _channels_to_label_mask(mask_channels):
    label_mask = np.zeros(mask_channels.shape[1:], dtype=np.uint8)

    for class_idx in range(mask_channels.shape[0]):
        label_mask[mask_channels[class_idx] > 0] = class_idx + 1

    return label_mask


def _build_color_mask(label_mask):
    safe_mask = np.clip(label_mask, 0, len(CLASS_COLORS) - 1)
    return CLASS_COLORS[safe_mask]


def _class_names_from_config(model_config):
    class_names = model_config.get("class_names", DEFAULT_AGHM_CONFIG["class_names"])
    return {int(key): value for key, value in class_names.items()}


def _safe_level_name(level_name):
    return str(level_name).replace("/", "-").replace("\\", "-").replace(" ", "_")


def _extract_dicom_from_slice_input(slice_input):
    if slice_input is None:
        return None

    if hasattr(slice_input, "pixel_array"):
        return slice_input

    if isinstance(slice_input, dict):
        return slice_input.get("dicom")

    if isinstance(slice_input, (list, tuple)):
        for item in slice_input:
            dicom_obj = _extract_dicom_from_slice_input(item)
            if dicom_obj is not None:
                return dicom_obj

    return None


def segment_axial_slice(
    dicom_slice,
    checkpoint_path=DEFAULT_AGHM_CHECKPOINT_PATH,
    output_dir="output/segmentation/axial",
    save_overlay=True,
    rotate_input_for_model=False,
    return_info=True,
    model_config=None,
    threshold=None,
    level_name=None,
    output_prefix=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = normalize_path(checkpoint_path)
    output_dir = normalize_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_norm, instance_number = _dicom_or_array_to_gray(dicom_slice)
    orig_h, orig_w = image_norm.shape[:2]
    final_image = (image_norm * 255).astype(np.uint8)

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

    if output_prefix is None:
        safe_level = _safe_level_name(level_name or "axial")
        output_prefix = f"{safe_level}_instance_{instance_number}"

    original_image_path = output_dir / f"{output_prefix}_original.png"
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
        if isinstance(logits, dict):
            logits = logits["logits"]
        prob_224 = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)

    pred_channels_224 = (prob_224 >= float(threshold)).astype(np.uint8)

    pred_channels_model_space = _resize_channels_nearest(
        pred_channels_224,
        (model_w, model_h),
    )

    if rotate_input_for_model:
        final_channels = np.stack(
            [
                inverse_rotate_from_model(channel)
                for channel in pred_channels_model_space
            ],
            axis=0,
        ).astype(np.uint8)
    else:
        final_channels = pred_channels_model_space

    if final_channels.shape[1:] != final_image.shape:
        final_channels = _resize_channels_nearest(final_channels, (orig_w, orig_h))

    final_mask = _channels_to_label_mask(final_channels)

    overlay_path = None

    if save_overlay:
        image_bgr = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
        overlay = image_bgr.copy().astype(np.float32)

        for class_idx in range(final_channels.shape[0]):
            class_mask = final_channels[class_idx] > 0
            if not np.any(class_mask):
                continue

            color = CLASS_COLORS[class_idx + 1].astype(np.float32)
            overlay[class_mask] = (
                overlay[class_mask] * SEGMENTATION_IMAGE_ALPHA
                + color * SEGMENTATION_MASK_ALPHA
            )

            kernel = np.ones((3, 3), dtype=np.uint8)
            eroded = cv2.erode(class_mask.astype(np.uint8), kernel, iterations=1) > 0
            border = class_mask ^ eroded
            overlay[border] = color

        overlay_path = output_dir / f"{output_prefix}_AGMH_overlay.png"
        cv2.imwrite(str(overlay_path), overlay.clip(0, 255).astype(np.uint8))

    result = {
        "mask": final_mask,
        "mask_channels": final_channels,
        "mask_224_channels": pred_channels_224,
        "prob_224": prob_224,
        "mask_model_space_channels": pred_channels_model_space,
        "image": final_image,
        "class_names": _class_names_from_config(model_config),
        "unique_classes": np.unique(final_mask).tolist(),
        "input_shape": (orig_h, orig_w),
        "model_input_shape": image_resized.shape,
        "output_shape": final_mask.shape,
        "rotate_input_for_model": rotate_input_for_model,
        "level_name": level_name,
        "instance_number": instance_number,
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


def axial_segmentation(
    slice_1=None,
    slice_2=None,
    slice_3=None,
    slice_4=None,
    slice_5=None,
    axial_slices=None,
    checkpoint_path=DEFAULT_AGHM_CHECKPOINT_PATH,
    output_dir="output/segmentation/axial",
    save_overlay=True,
    rotate_input_for_model=False,
    return_info=True,
    model_config=None,
    threshold=None,
    level_names=None,
):
    if axial_slices is None:
        axial_slices = [slice_1, slice_2, slice_3, slice_4, slice_5]

    axial_slices = list(axial_slices)
    level_names = list(level_names or DEFAULT_LEVEL_NAMES)
    model_config = model_config or load_aghm_config()
    output_dir = normalize_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for idx, slice_input in enumerate(axial_slices):
        level_name = level_names[idx] if idx < len(level_names) else f"slice_{idx + 1}"
        dicom_slice = _extract_dicom_from_slice_input(slice_input)

        if dicom_slice is None:
            results.append({
                "level_index": idx + 1,
                "level_name": level_name,
                "error": "No DICOM slice found in axial slice input.",
            })
            continue

        output_prefix = f"axial_{idx + 1}_{_safe_level_name(level_name)}"

        result = segment_axial_slice(
            dicom_slice=dicom_slice,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            save_overlay=save_overlay,
            rotate_input_for_model=rotate_input_for_model,
            return_info=True,
            model_config=model_config,
            threshold=threshold,
            level_name=level_name,
            output_prefix=output_prefix,
        )

        result["level_index"] = idx + 1
        results.append(result)

    if return_info:
        return {
            "results": results,
            "masks": [
                item["mask"]
                for item in results
                if "mask" in item
            ],
            "level_names": level_names[: len(axial_slices)],
            "checkpoint_path": str(normalize_path(checkpoint_path)),
            "config_path": str(DEFAULT_AGHM_CONFIG_PATH),
            "output_dir": str(output_dir),
            "model_name": "AGMH",
            "threshold": float(threshold if threshold is not None else model_config.get("threshold", 0.5)),
        }

    return [
        item["mask"]
        for item in results
        if "mask" in item
    ]
