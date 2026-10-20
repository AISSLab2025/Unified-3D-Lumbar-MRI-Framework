import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from tools.classification.models.sagittal_model import SwinWithCAMUncertainty

# Module-level flag to ensure classification output directory is cleaned only once per run
_CLASSIFICATION_DIR_CLEANED = False


def normalize_path(path_str: Any) -> Path:
    if not path_str:
        return Path()
    path_str = str(path_str)
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        path_str = "C:/" + path_str[7:]
    elif os.name != "nt" and (path_str.startswith("C:/") or path_str.startswith("C:\\")):
        path_str = "/mnt/c/" + path_str[3:].replace("\\", "/")
    return Path(path_str)


def generate_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate Grad-CAM from model.cam layer.
    image_tensor shape: [1, 3, H, W]
    """

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = model.cam.register_forward_hook(forward_hook)
    backward_handle = model.cam.register_full_backward_hook(backward_hook)

    outputs = model(image_tensor)
    logits = outputs["final_logits"]

    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()

    score = logits[:, target_class]

    model.zero_grad()
    score.backward()

    acts = activations[0]      # [1, C, H, W]
    grads = gradients[0]       # [1, C, H, W]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)

    cam = F.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    forward_handle.remove()
    backward_handle.remove()

    return cam, outputs


def recover_cam_to_original_roi(
    cam: np.ndarray,
    cropped_roi: Image.Image,
    processed_roi: np.ndarray
) -> np.ndarray:
    """
    Recover CAM from classifier space to the original cropped ROI dimensions.
    
    1. Resize the CAM (feature map) to processed crop space (classifier input shape).
    2. Infer scale and padding used in preprocessing.
    3. Remove the padding from the CAM.
    4. Resize the cropped CAM back to original ROI dimensions.
    """
    # 1. Resize CAM to processed crop space (classifier input shape)
    proc_h, proc_w = processed_roi.shape[:2]
    cam_resized = cv2.resize(cam, (proc_w, proc_h))

    # 2. Get original dimensions from cropped_roi PIL image
    orig_w, orig_h = cropped_roi.size

    # 3. Re-calculate scale and padding as in tools/roi/postprocessing.py
    scale = min(float(proc_w) / float(orig_w), float(proc_h) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    pad_top = (proc_h - new_h) // 2
    pad_left = (proc_w - new_w) // 2

    # 4. Remove padding from the CAM
    y_start = max(0, pad_top)
    y_end = min(proc_h, pad_top + new_h)
    x_start = max(0, pad_left)
    x_end = min(proc_w, pad_left + new_w)

    cam_cropped = cam_resized[y_start:y_end, x_start:x_end]

    # 5. Resize back to original ROI dimensions (orig_w, orig_h)
    cam_recovered = cv2.resize(cam_cropped, (orig_w, orig_h))

    # Re-normalize to [0, 1] range to ensure visualization quality
    cam_recovered = cam_recovered - cam_recovered.min()
    cam_recovered = cam_recovered / (cam_recovered.max() + 1e-8)

    return cam_recovered


def overlay_cam_on_roi(
    cropped_roi: Image.Image,
    cam: np.ndarray
) -> np.ndarray:
    """
    Overlay the recovered CAM on the original ROI crop image.
    cropped_roi: PIL Image representing the original crop
    cam: numpy heatmap of shape (orig_h, orig_w) with values in [0, 1]
    """
    roi_np = np.array(cropped_roi)
    if roi_np.dtype != np.uint8:
        roi_np = roi_np.astype(np.uint8)

    # Convert RGB/Grayscale to BGR (Opencv standard)
    if len(roi_np.shape) == 2:
        roi_bgr = cv2.cvtColor(roi_np, cv2.COLOR_GRAY2BGR)
    elif roi_np.shape[2] == 3:
        roi_bgr = cv2.cvtColor(roi_np, cv2.COLOR_RGB2BGR)
    else:
        roi_bgr = roi_np.copy()

    # Apply JET color map to the CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Blend original ROI crop and heatmap
    overlay = cv2.addWeighted(roi_bgr, 0.45, heatmap, 0.40, 0)
    return overlay


def save_gradcam_overlay(original_crop: np.ndarray, cam: np.ndarray, save_path: Path) -> None:
    """
    Save Grad-CAM overlay image. (Maintained for backward compatibility)
    original_crop: numpy image, H,W or H,W,3
    cam: numpy heatmap
    """
    if original_crop.dtype != np.uint8:
        original_crop = original_crop.astype(np.uint8)

    if len(original_crop.shape) == 2:
        original_crop = cv2.cvtColor(original_crop, cv2.COLOR_GRAY2BGR)
    else:
        original_crop = cv2.cvtColor(original_crop, cv2.COLOR_RGB2BGR)

    h, w = original_crop.shape[:2]
    cam = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(original_crop, 0.45, heatmap, 0.40, 0)

    cv2.imwrite(str(save_path), overlay)


def load_original_slice_from_dicom(
    instance_number: str,
    workspace_root: Path
) -> Optional[np.ndarray]:
    """
    Find and load the original sagittal DICOM slice with the matching InstanceNumber.
    """
    import pydicom
    workspace_root = normalize_path(workspace_root)
    # Walk the directories under workspace_root
    for root, dirs, files in os.walk(str(workspace_root)):
        # Skip output directory and checkpoints to avoid reading wrong files
        if "output" in root or "checkpoints" in root or ".gemini" in root:
            continue
        
        # Only search inside sagittal folders
        if "sagittal" not in root.lower():
            continue
        
        # Look for DICOM files
        for f in files:
            if f.lower().endswith(".dcm") or f.lower().startswith("im"):
                dcm_path = Path(root) / f
                try:
                    # Read metadata only to check InstanceNumber first (fast)
                    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                    if str(getattr(ds, "InstanceNumber", "")) == str(instance_number):
                        # Read the full DICOM file
                        ds_full = pydicom.dcmread(str(dcm_path))
                        arr = ds_full.pixel_array
                        
                        # Process pixel array to match detector's PIL creation
                        if arr.ndim == 3 and arr.shape[0] in (1, 3):
                            arr = np.transpose(arr, (1, 2, 0))
                        
                        # Convert to PIL and then back to BGR numpy array
                        pil = Image.fromarray(arr).convert("RGB")
                        arr_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                        return arr_bgr
                except Exception:
                    # Ignore corrupted or invalid files
                    continue
    return None


def Sagittal_ROI_classification(
    roi_list: List[Dict[str, Any]],
    output_dir: Optional[Any] = None,
    config_path: Any = "checkpoints/classification/Sagittal_foramina_config.json",
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Input:
        roi_list: list of ROI dictionaries from Sagittal_ROI_detection
                  each ROI must contain: level, bbox, processed_roi, confidence, cropped_roi, overlay_image_path

    Output:
        results: list of dictionaries with classification result for each ROI
    """
    config_path = normalize_path(config_path)

    if output_dir is None:
        found_output_dir = None
        for r in roi_list:
            overlay_path_str = r.get("overlay_image_path")
            if overlay_path_str:
                try:
                    overlay_path = Path(overlay_path_str)
                    found_output_dir = overlay_path.parent.parent / "classification"
                    break
                except Exception:
                    pass
        if found_output_dir is not None:
            output_dir = found_output_dir
        else:
            output_dir = Path("output/classification")

    output_dir = normalize_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    model_path = normalize_path(config["model_path"])

    num_classes = config["num_classes"]
    class_names = config["class_names"]
    img_size = config["img_size"]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])

    model = SwinWithCAMUncertainty(
        num_classes=num_classes,
        dropout_p=config["dropout_p"],
        cam_reduction=config["cam_reduction"],
        attention_mechanism=config["attention_mechanism"],
        attention_stage=config["attention_stage"],
        ugr_hidden_dim=config["ugr_hidden_dim"],
        ugr_dropout=config["ugr_dropout"],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # --- DELETE PREVIOUS SAVED GRAD-CAM RESULTS IN THE ACTIVE OUTPUT DIRECTORY ---
    global _CLASSIFICATION_DIR_CLEANED
    if not _CLASSIFICATION_DIR_CLEANED:
        target_delete_dir = output_dir
        for d in [target_delete_dir, output_dir]:
            if d.exists():
                for item in d.iterdir():
                    if item.is_file():
                        try:
                            item.unlink()
                        except Exception as e:
                            print(f"Warning: Could not delete previous result {item}: {e}")
        _CLASSIFICATION_DIR_CLEANED = True

    results = []
    total_classification_time = 0.0

    # Dictionaries to track base image slices and output paths
    slice_visualizations: Dict[Path, Optional[np.ndarray]] = {}
    slice_cam_paths: Dict[Path, Path] = {}

    for roi in roi_list:
        crop = roi["processed_roi"]

        if crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)

        image = Image.fromarray(crop)
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Grad-CAM needs gradient, so DO NOT use torch.no_grad()
        start_cls = time.perf_counter()
        cam, outputs = generate_gradcam(model, image_tensor)
        total_classification_time += time.perf_counter() - start_cls

        logits = outputs["final_logits"]
        probs = torch.softmax(logits, dim=1)[0]

        pred_idx = torch.argmax(probs).item()
        pred_prob = probs[pred_idx].item()

        level = roi["level"]

        # Handle image loading and output path mapping
        overlay_image_path_str = roi.get("overlay_image_path")

        if overlay_image_path_str:
            normalized_overlay_path = normalize_path(overlay_image_path_str)

            # Load the original clean slice image from DICOM only once
            if normalized_overlay_path not in slice_visualizations:
                # Direct lookup using the stored dicom_path from the detector
                dicom_path_str = roi.get("dicom_path")
                slice_img = None
                if dicom_path_str:
                    try:
                        import pydicom
                        normalized_dicom_path = normalize_path(dicom_path_str)
                        if normalized_dicom_path.exists():
                            ds_full = pydicom.dcmread(str(normalized_dicom_path))
                            arr = ds_full.pixel_array
                            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                                arr = np.transpose(arr, (1, 2, 0))
                            pil = Image.fromarray(arr).convert("RGB")
                            slice_img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        print(f"Warning: Failed to load DICOM directly from {dicom_path_str}: {e}")

                # Fallback to search method if direct lookup is not available or failed
                if slice_img is None:
                    workspace_root = Path(__file__).resolve().parent.parent.parent
                    instance_num = roi.get("instance_number")
                    slice_img = load_original_slice_from_dicom(instance_num, workspace_root)
                
                if slice_img is None:
                    print(f"Warning: Could not find original DICOM for instance {roi.get('instance_number')}. Falling back to overlay image.")
                    if normalized_overlay_path.exists():
                        slice_img = cv2.imread(str(normalized_overlay_path))
                    else:
                        slice_img = None
                
                slice_visualizations[normalized_overlay_path] = slice_img
            else:
                slice_img = slice_visualizations[normalized_overlay_path]

            # Map to shared output Grad-CAM path
            if normalized_overlay_path not in slice_cam_paths:
                slice_stem = normalized_overlay_path.stem
                gradcam_path = output_dir / f"{slice_stem}_gradcam.png"
                slice_cam_paths[normalized_overlay_path] = gradcam_path
            else:
                gradcam_path = slice_cam_paths[normalized_overlay_path]
        else:
            slice_img = None
            gradcam_path = output_dir / f"level_{level}_gradcam.png"

        # Recover original geometry
        cam_recovered = recover_cam_to_original_roi(cam, roi["cropped_roi"], roi["processed_roi"])

        # Overlay recovered CAM on the original ROI crop
        overlay_roi = overlay_cam_on_roi(roi["cropped_roi"], cam_recovered)

        # Paste overlayed ROI back into original sagittal image
        if slice_img is not None:
            x1, y1, x2, y2 = roi["bbox"]
            h_img, w_img = slice_img.shape[:2]
            x1i = int(max(0, round(x1)))
            y1i = int(max(0, round(y1)))
            x2i = int(min(w_img, round(x2)))
            y2i = int(min(h_img, round(y2)))

            crop_h = y2i - y1i
            crop_w = x2i - x1i

            if overlay_roi.shape[0] != crop_h or overlay_roi.shape[1] != crop_w:
                overlay_roi = cv2.resize(overlay_roi, (crop_w, crop_h))

            slice_img[y1i:y2i, x1i:x2i] = overlay_roi

            # Determine colors
            class_name = class_names[pred_idx]
            if class_name == "Normal":
                color_bgr = (0, 255, 0)
            elif class_name == "Stenosis":
                color_bgr = (0, 255, 255) # Yellow (BGR)
            elif class_name == "Severe_Stenosis":
                color_bgr = (0, 0, 255) # Red (BGR)
            else:
                color_bgr = (255, 255, 255)

            # Draw the boundary box in the class color with thickness=2
            cv2.rectangle(slice_img, (x1i, y1i), (x2i, y2i), color_bgr, thickness=2)
            slice_visualizations[normalized_overlay_path] = slice_img
            
        results.append({
            "level": level,
            "level_name": roi.get("level_name"),   # <-- Add this

            "bbox": roi["bbox"],
            "detection_confidence": roi["confidence"],

            "predicted_index": pred_idx,
            "predicted_class": class_names[pred_idx],
            "predicted_probability": pred_prob,

            "probabilities": {
                class_names[i]: probs[i].item()
                for i in range(num_classes)
            },

            "uncertainty": outputs["uncertainty"][0].item(),

            "gradcam_path": str(gradcam_path),
        })

    # Save only one Grad-CAM image per sagittal slice
    for normalized_overlay_path, slice_img in slice_visualizations.items():
        if slice_img is not None:
            gradcam_path = slice_cam_paths[normalized_overlay_path]
            cv2.imwrite(str(gradcam_path), slice_img)

    print(f"\n[TIMING] Sagittal Foraminal Classification Time: {total_classification_time:.4f} seconds")

    return results
