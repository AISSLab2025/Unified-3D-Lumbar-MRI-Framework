import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# =========================================================
# Path Normalization Helper
# =========================================================
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


# =========================================================
# DICOM to uint8
# =========================================================
def dicom_to_uint8(dicom):
    img = dicom.pixel_array.astype(np.float32)
    img = img - np.min(img)
    img = img / (np.max(img) + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img


# =========================================================
# CAM Recovery & Blending
# =========================================================
def recover_cam_to_expanded_crop(
    cam: np.ndarray,
    expanded_crop: np.ndarray,
    processed_roi: np.ndarray
) -> np.ndarray:
    proc_h, proc_w = processed_roi.shape[:2]
    cam_resized = cv2.resize(cam, (proc_w, proc_h))

    orig_h, orig_w = expanded_crop.shape[:2]

    scale = min(float(proc_w) / float(orig_w), float(proc_h) / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    pad_top = (proc_h - new_h) // 2
    pad_left = (proc_w - new_w) // 2

    y_start = max(0, pad_top)
    y_end = min(proc_h, pad_top + new_h)
    x_start = max(0, pad_left)
    x_end = min(proc_w, pad_left + new_w)

    cam_cropped = cam_resized[y_start:y_end, x_start:x_end]
    cam_recovered = cv2.resize(cam_cropped, (orig_w, orig_h))

    cam_recovered = cam_recovered - cam_recovered.min()
    cam_recovered = cam_recovered / (cam_recovered.max() + 1e-8)

    return cam_recovered


def overlay_cam_on_roi(
    cropped_roi: np.ndarray,
    cam: np.ndarray
) -> np.ndarray:
    roi_np = cropped_roi.copy()
    if roi_np.dtype != np.uint8:
        roi_np = roi_np.astype(np.uint8)

    if len(roi_np.shape) == 2:
        roi_bgr = cv2.cvtColor(roi_np, cv2.COLOR_GRAY2BGR)
    elif roi_np.shape[2] == 3:
        roi_bgr = cv2.cvtColor(roi_np, cv2.COLOR_RGB2BGR)
    else:
        roi_bgr = roi_np.copy()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(roi_bgr, 0.45, heatmap, 0.40, 0)
    return overlay


# =========================================================
# Combined Visualization Generator
# =========================================================
def save_combined_axial_slices(
    foraminal_results,
    lateral_results,
    central_results,
    save_dir
):
    """
    Combines foraminal, lateral, and central axial classification overlays
    into a single slice per level (e.g. L4-L5_merged.png).
    """
    save_dir = normalize_path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_images = {}

    # Group classification results by level name
    results_by_level = defaultdict(list)
    for r in foraminal_results:
        results_by_level[r["level_name"]].append(r)
    for r in lateral_results:
        results_by_level[r["level_name"]].append(r)
    for r in central_results:
        results_by_level[r["level_name"]].append(r)

    # Region order helper for sorting ROIs: LFS, RFS, LLS, RLS, SCS
    def get_region_order(item):
        rname = item["region_name"]
        order = {"LFS": 0, "RFS": 1, "LLS": 2, "RLS": 3, "SCS": 4}
        return order.get(rname, 5)

    # Font setup
    if os.name == "nt":
        font_dir = Path("C:/Windows/Fonts")
    else:
        font_dir = Path("/mnt/c/Windows/Fonts")

    title_font = None
    number_font = None

    for font_name in ["calibrib.ttf", "calibri.ttf", "arialbd.ttf", "arial.ttf"]:
        try_path = font_dir / font_name
        if try_path.exists():
            try:
                title_font = ImageFont.truetype(str(try_path), 22)
                number_font = ImageFont.truetype(str(try_path), 18)
                break
            except Exception:
                pass
        try:
            title_font = ImageFont.truetype(font_name, 22)
            number_font = ImageFont.truetype(font_name, 18)
            break
        except Exception:
            pass

    if title_font is None:
        try:
            title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
            number_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except Exception:
            try:
                title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
                number_font = ImageFont.truetype("DejaVuSans.ttf", 18)
            except Exception:
                title_font = ImageFont.load_default()
                number_font = ImageFont.load_default()

    # Generate combined slice per level
    for level_name, slice_items in results_by_level.items():
        if not slice_items:
            continue

        safe_level_name = level_name.replace("/", "-")
        merged_path = save_dir / f"{safe_level_name}_merged.png"

        # Load clean base slice image using first item's dicom
        first_item = slice_items[0]
        slice_img = None
        if "dicom" in first_item and first_item["dicom"] is not None:
            try:
                full_image = dicom_to_uint8(first_item["dicom"])
                slice_img = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(f"Warning: Failed to load DICOM from first_item['dicom']: {e}")

        # Fallback to load from path
        if slice_img is None:
            # We can try to load via pydicom if dicom_path was stored
            dicom_path_str = first_item.get("dicom_path")
            if dicom_path_str:
                try:
                    import pydicom
                    normalized_dicom_path = normalize_path(dicom_path_str)
                    if normalized_dicom_path.exists():
                        ds_full = pydicom.dcmread(str(normalized_dicom_path))
                        full_image = dicom_to_uint8(ds_full)
                        slice_img = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)
                except Exception as e:
                    print(f"Warning: Failed to load DICOM from path {dicom_path_str}: {e}")

        if slice_img is None:
            slice_img = np.zeros((512, 512, 3), dtype=np.uint8)

        h_img, w_img = slice_img.shape[:2]

        # Convert clean slice to PIL for all drawing
        slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
        pil_slice = Image.fromarray(slice_img_rgb)
        draw = ImageDraw.Draw(pil_slice)

        # Sort the items to ensure consistent numbering: LFS, RFS, LLS, RLS, SCS
        sorted_items = sorted(slice_items, key=get_region_order)
        legend_lines = []

        for idx, item in enumerate(sorted_items, start=1):
            cam = item.get("cam")
            expanded_crop = item.get("expanded_crop")
            processed_roi = item.get("processed_roi")
            xmin, ymin, xmax, ymax = item["original_bbox"]
            ex_xmin, ex_ymin, ex_xmax, ex_ymax = item["expanded_bbox"]
            region_name = item["region_name"]
            predicted_class = item["predicted_class"]
            predicted_probability = item["predicted_probability"]
            det_conf = item.get("detection_confidence", 0.0)

            # Skip drawing if any raw arrays are missing
            if cam is None or expanded_crop is None or processed_roi is None:
                continue

            # Recover CAM
            cam_recovered = recover_cam_to_expanded_crop(cam, expanded_crop, processed_roi)

            # Crop CAM and image to original bbox region
            x1_crop = int(max(0, round(xmin - ex_xmin)))
            y1_crop = int(max(0, round(ymin - ex_ymin)))
            x2_crop = int(min(expanded_crop.shape[1], round(xmax - ex_xmin)))
            y2_crop = int(min(expanded_crop.shape[0], round(ymax - ex_ymin)))

            original_crop = expanded_crop[y1_crop:y2_crop, x1_crop:x2_crop]
            cam_original = cam_recovered[y1_crop:y2_crop, x1_crop:x2_crop]

            # Blend CAM
            overlay_roi = overlay_cam_on_roi(original_crop, cam_original)

            # Paste blended ROI back
            x1i = int(max(0, round(xmin)))
            y1i = int(max(0, round(ymin)))
            x2i = int(min(w_img, round(xmax)))
            y2i = int(min(h_img, round(ymax)))

            crop_h = y2i - y1i
            crop_w = x2i - x1i

            if overlay_roi.shape[0] != crop_h or overlay_roi.shape[1] != crop_w:
                overlay_roi = cv2.resize(overlay_roi, (crop_w, crop_h))

            overlay_roi_rgb = cv2.cvtColor(overlay_roi, cv2.COLOR_BGR2RGB)
            roi_pil = Image.fromarray(overlay_roi_rgb)
            pil_slice.paste(roi_pil, (x1i, y1i))

            # Select class color
            if predicted_class == "Normal":
                color_rgb = (0, 255, 0)
            elif predicted_class == "Stenosis":
                color_rgb = (255, 255, 0)
            elif predicted_class == "Severe_Stenosis":
                color_rgb = (255, 0, 0)
            else:
                color_rgb = (255, 255, 255)

            # Draw bounding box (thickness=1)
            draw.rectangle([x1i, y1i, x2i, y2i], outline=color_rgb, width=1)

            # Draw number above box with black background
            number_text = str(idx)
            bbox = draw.textbbox((0, 0), number_text, font=number_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            text_x = (x1i + x2i) / 2 - text_w / 2
            text_y = y1i - text_h - 4
            if text_y < 0:
                text_y = y1i + 2

            draw.rectangle(
                [
                    text_x - 2,
                    text_y - 2,
                    text_x + text_w + 2,
                    text_y + text_h + 2
                ],
                fill=(0, 0, 0)
            )

            draw.text(
                (text_x, text_y),
                number_text,
                fill=color_rgb,
                font=number_font
            )

            # Legend entry
            class_label = predicted_class.replace('_', ' ')
            legend_text = f"{idx} - {region_name} ({det_conf:.2f}) {class_label} ({predicted_probability:.2f})"
            legend_lines.append((legend_text, color_rgb))

        # Draw level name in top-left corner
        draw.text((21, 21), level_name, fill=(0, 0, 0), font=title_font)
        draw.text((20, 20), level_name, fill=(255, 255, 255), font=title_font)

        # Draw Legend entries below title
        start_x = 20
        start_y = 55
        line_spacing = 22
        for i, (text, col_rgb) in enumerate(legend_lines):
            draw.text(
                (start_x, start_y + i * line_spacing),
                text,
                fill=col_rgb,
                font=number_font
            )

        # Save merged image
        slice_img = cv2.cvtColor(np.array(pil_slice), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(merged_path), slice_img)
        saved_images[level_name] = str(merged_path)
        print(f"Saved merged slice: {merged_path}")

    return saved_images
