from tools.roi.roi_timing import ROITiming
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import os

from PIL import ImageDraw, ImageFont, Image
from tools.roi.sagittal_roi_detector import normalize_path
from tools.roi.coordinate_csv import append_coordinate_csv, roi_to_coordinate_row


REGION_TO_ID = {
    "LFS": 0,
    "LLS": 1,
    "SCS": 2,
    "RLS": 3,
    "RFS": 4,
}

EXPECTED_REGIONS = {"LFS", "LLS", "SCS", "RLS", "RFS"}
AXIAL_COMBINED_ROI_NUMBERS = {
    "LFS": "1",
    "RFS": "2",
    "LLS": "3",
    "RLS": "4",
    "SCS": "5",
}

AXIAL_ROI_DRAW_COLOR = (255, 255, 0)


def load_number_font(font_size=18):
    font_dir = Path("C:/Windows/Fonts") if os.name == "nt" else Path("/mnt/c/Windows/Fonts")
    for font_name in ("calibrib.ttf", "calibri.ttf", "arialbd.ttf", "arial.ttf"):
        font_path = font_dir / font_name
        try:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), font_size)
        except Exception:
            pass

    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except Exception:
            pass

    return ImageFont.load_default()


# =========================================================
# Load reference tables
# =========================================================
def load_reference_tables(
    table_dir="."
):
    table_dir = normalize_path(table_dir)

    offset_csv = table_dir / "average_roi_pairwise_offsets_mm.csv"
    size_csv = table_dir / "average_roi_size_mm.csv"

    offset_df = pd.read_csv(offset_csv)
    size_df = pd.read_csv(size_csv)

    offset_table = {}
    for _, row in offset_df.iterrows():
        ref = row["reference_roi"]
        target = row["target_roi"]

        offset_table[(ref, target)] = {
            "dx_mm": float(row["mean_dx_mm"]),
            "dy_mm": float(row["mean_dy_mm"]),
            "count": int(row["count"]),
        }

    size_table = {}
    for _, row in size_df.iterrows():
        roi = row["roi"]

        size_table[roi] = {
            "width_mm": float(row["mean_width_mm"]),
            "height_mm": float(row["mean_height_mm"]),
            "count": int(row["count"]),
        }

    return offset_table, size_table


# =========================================================
# DICOM image utilities
# =========================================================
def dicom_to_pil(dicom_slice):
    img = dicom_slice.pixel_array.astype(np.float32)

    img -= img.min()

    if img.max() > 0:
        img /= img.max()

    img = (img * 255).astype(np.uint8)

    return Image.fromarray(img).convert("RGB")


def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return cx, cy


def bbox_from_center(cx, cy, width_px, height_px, image_w, image_h):
    x1 = cx - width_px / 2.0
    y1 = cy - height_px / 2.0
    x2 = cx + width_px / 2.0
    y2 = cy + height_px / 2.0

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w, x2))
    y2 = max(0, min(image_h, y2))

    return [float(x1), float(y1), float(x2), float(y2)]


# =========================================================
# DICOM coordinate conversion
# =========================================================
def pixel_to_patient_coordinate(dicom_slice, x, y):
    """
    Convert image pixel coordinate to DICOM patient coordinate.

    DICOM uses LPS coordinate system:
        +X = patient left
        +Y = patient posterior
        +Z = patient superior

    x = image column coordinate
    y = image row coordinate
    """

    ipp = np.array(dicom_slice.ImagePositionPatient, dtype=np.float64)
    iop = np.array(dicom_slice.ImageOrientationPatient, dtype=np.float64)

    row_direction = iop[:3]
    col_direction = iop[3:]

    pixel_spacing = getattr(dicom_slice, "PixelSpacing", [1.0, 1.0])
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    patient_coord = (
        ipp
        + row_direction * x * col_spacing
        + col_direction * y * row_spacing
    )

    return patient_coord


def get_roi_patient_x(dicom_slice, roi):
    """
    Return patient-left coordinate of ROI center.
    Larger X means more toward patient's LEFT.
    Smaller X means more toward patient's RIGHT.
    """

    cx, cy = get_bbox_center(roi["bbox"])
    patient_coord = pixel_to_patient_coordinate(dicom_slice, cx, cy)

    return float(patient_coord[0])


# =========================================================
# Rename left/right regions using DICOM metadata
# =========================================================
def rename_left_right_regions_using_dicom(slice_rois):
    """
    Correct L/R labels using DICOM patient coordinates.

    SCS is used as the center reference.
    Any foraminal ROI:
        left side  -> LFS
        right side -> RFS

    Any lateral ROI:
        left side  -> LLS
        right side -> RLS

    SCS stays SCS.
    """

    if len(slice_rois) == 0:
        return slice_rois

    dicom_slice = slice_rois[0]["dicom"]

    scs_rois = [
        roi for roi in slice_rois
        if roi["region_name"] == "SCS"
    ]

    if len(scs_rois) > 0:
        # use highest confidence SCS if multiple exist
        scs_roi = max(
            scs_rois,
            key=lambda r: float(r.get("confidence", 0.0))
        )
        center_x = get_roi_patient_x(dicom_slice, scs_roi)
    else:
        # fallback: use image center if SCS missing
        pil_img = dicom_to_pil(dicom_slice)
        image_w, image_h = pil_img.size

        center_coord = pixel_to_patient_coordinate(
            dicom_slice,
            image_w / 2.0,
            image_h / 2.0
        )
        center_x = float(center_coord[0])

    updated_rois = []

    for roi in slice_rois:
        roi = dict(roi)

        old_region = roi["region_name"]

        if old_region == "SCS":
            roi["region_name"] = "SCS"
            roi["region"] = REGION_TO_ID["SCS"]
            roi["side_corrected_by_dicom"] = False
            updated_rois.append(roi)
            continue

        roi_x = get_roi_patient_x(dicom_slice, roi)

        if roi_x > center_x:
            anatomical_side = "left"
        else:
            anatomical_side = "right"

        # Foraminal regions
        if old_region in ["LFS", "RFS"]:
            new_region = "LFS" if anatomical_side == "left" else "RFS"

        # Lateral recess regions
        elif old_region in ["LLS", "RLS"]:
            new_region = "LLS" if anatomical_side == "left" else "RLS"

        else:
            new_region = old_region

        roi["original_region_name"] = old_region
        roi["region_name"] = new_region
        roi["region"] = REGION_TO_ID[new_region]
        roi["patient_x"] = roi_x
        roi["reference_center_x"] = center_x
        roi["anatomical_side"] = anatomical_side
        roi["side_corrected_by_dicom"] = old_region != new_region

        updated_rois.append(roi)

    updated_rois = sorted(updated_rois, key=lambda x: x["region"])

    return updated_rois


# =========================================================
# Remove duplicates after DICOM correction
# =========================================================
def keep_highest_confidence_per_region(slice_rois):
    """
    If more than one ROI exists for the same corrected region,
    keep only the highest-confidence ROI.
    """

    best_rois = {}

    for roi in slice_rois:
        region_name = roi["region_name"]
        confidence = float(roi.get("confidence", 0.0))

        if region_name not in best_rois:
            best_rois[region_name] = roi
        else:
            old_confidence = float(
                best_rois[region_name].get("confidence", 0.0)
            )

            if confidence > old_confidence:
                best_rois[region_name] = roi

    return sorted(best_rois.values(), key=lambda x: x["region"])


# =========================================================
# Estimation weights
# =========================================================
def get_roi_weight(roi):
    if roi.get("estimated", False):
        return float(roi.get("estimation_weight", 0.3))

    return float(roi.get("confidence", 0.1))


# =========================================================
# Estimate one missing ROI
# =========================================================
def estimate_one_missing_roi(
    dicom_slice,
    existing_rois,
    missing_region,
    offset_table,
    size_table
):
    pil_img = dicom_to_pil(dicom_slice)
    image_w, image_h = pil_img.size

    pixel_spacing = getattr(dicom_slice, "PixelSpacing", [1.0, 1.0])
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])

    candidate_centers = []
    candidate_weights = []
    used_reference_regions = []

    for ref_roi in existing_rois:
        ref_region = ref_roi["region_name"]

        if ref_region == missing_region:
            continue

        if (ref_region, missing_region) not in offset_table:
            continue

        ref_cx, ref_cy = get_bbox_center(ref_roi["bbox"])

        offset_info = offset_table[(ref_region, missing_region)]

        dx_mm = float(offset_info["dx_mm"])
        dy_mm = float(offset_info["dy_mm"])

        dx_px = dx_mm / col_spacing
        dy_px = dy_mm / row_spacing

        target_cx = ref_cx - dx_px
        target_cy = ref_cy - dy_px

        weight = get_roi_weight(ref_roi)

        if weight <= 0:
            continue

        candidate_centers.append([target_cx, target_cy])
        candidate_weights.append(weight)
        used_reference_regions.append(ref_region)

    if len(candidate_centers) == 0:
        raise ValueError(
            f"Cannot estimate {missing_region}: no valid reference ROI found"
        )

    candidate_centers = np.array(candidate_centers, dtype=np.float32)
    candidate_weights = np.array(candidate_weights, dtype=np.float32)

    raw_weights = candidate_weights.copy()

    candidate_weights = candidate_weights / (
        candidate_weights.sum() + 1e-8
    )

    target_cx = float(
        np.sum(candidate_centers[:, 0] * candidate_weights)
    )
    target_cy = float(
        np.sum(candidate_centers[:, 1] * candidate_weights)
    )

    width_mm = size_table[missing_region]["width_mm"]
    height_mm = size_table[missing_region]["height_mm"]

    width_px = width_mm / col_spacing
    height_px = height_mm / row_spacing

    estimated_bbox = bbox_from_center(
        cx=target_cx,
        cy=target_cy,
        width_px=width_px,
        height_px=height_px,
        image_w=image_w,
        image_h=image_h
    )

    x1, y1, x2, y2 = [int(round(v)) for v in estimated_bbox]

    cropped_roi = pil_img.crop((x1, y1, x2, y2))

    estimated_roi = {
        "region": REGION_TO_ID[missing_region],
        "region_name": missing_region,
        "bbox": estimated_bbox,
        "cropped_roi": cropped_roi,
        "confidence": 0.0,
        "instance_number": int(getattr(dicom_slice, "InstanceNumber", -1)),
        "dicom": dicom_slice,
        "estimated": True,
        "reference_region": "multi_roi_weighted_dx_dy",
        "num_reference_rois": len(candidate_centers),
        "used_reference_regions": used_reference_regions,
        "raw_reference_weights": raw_weights.tolist(),
        "normalized_reference_weights": candidate_weights.tolist(),
        "estimation_weight": 0.3,
    }

    return estimated_roi


# =========================================================
# Estimate missing axial ROIs
# =========================================================
def estimate_missing_axial_rois(
    dicom_slice,
    existing_rois,
    missing_regions,
    offset_table,
    size_table
):
    updated_rois = list(existing_rois)

    missing_regions_sorted = sorted(
        list(missing_regions),
        key=lambda x: 1 if x == "SCS" else 0
    )

    for missing_region in missing_regions_sorted:
        estimated_roi = estimate_one_missing_roi(
            dicom_slice=dicom_slice,
            existing_rois=updated_rois,
            missing_region=missing_region,
            offset_table=offset_table,
            size_table=size_table
        )

        updated_rois.append(estimated_roi)

    updated_rois = sorted(updated_rois, key=lambda x: x["region"])

    return updated_rois




def save_axial_roi_debug_images(slice_rois, save_dir):

    save_dir = normalize_path(save_dir)

 

    save_dir.mkdir(parents=True, exist_ok=True)

    if len(slice_rois) == 0:
        return

    # --------------------------------------------------
    # Image
    # --------------------------------------------------
    dicom_slice = slice_rois[0]["dicom"]
    instance_number = slice_rois[0]["instance_number"]

    pil_img = dicom_to_pil(dicom_slice)

    overlay = pil_img.copy()
    draw = ImageDraw.Draw(overlay)
    font = load_number_font()

    # --------------------------------------------------
    # Settings
    # --------------------------------------------------
    roi_color = AXIAL_ROI_DRAW_COLOR

    sorted_rois = sorted(
        slice_rois,
        key=lambda x: x["region"]
    )

    # --------------------------------------------------
    # Draw ROIs
    # --------------------------------------------------
    for roi in sorted_rois:

        x1, y1, x2, y2 = roi["bbox"]
        region_name = roi["region_name"]
        number_text = AXIAL_COMBINED_ROI_NUMBERS.get(region_name, str(roi["region"] + 1))

        # Bounding box
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=roi_color,
            width=1
        )
        text_bbox = draw.textbbox((0, 0), number_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = (x1 + x2) / 2 - text_w / 2
        text_y = y1 - text_h - 4
        if text_y < 0:
            text_y = y1 + 2

        draw.rectangle(
            [
                text_x - 2,
                text_y - 2,
                text_x + text_w + 2,
                text_y + text_h + 2,
            ],
            fill=(0, 0, 0),
        )
        draw.text((text_x, text_y), number_text, fill=roi_color, font=font)

    # --------------------------------------------------
    # Save overlay only
    # --------------------------------------------------
    output_path = (
        save_dir /
        f"slice_{instance_number}.png"
    )

    overlay.save(output_path)

    print(f"Saved overlay: {output_path}")
    return output_path # <--- ADD THIS LINE



def axial_coordinates_csv_path(output_dir):
    output_dir = Path(output_dir)

    if output_dir.name == "axial" and output_dir.parent.name == "ROIs_detection":
        return output_dir.parent.parent / "axial_coordinates.csv"

    return output_dir / "axial_coordinates.csv"






# =========================================================
# Main function
# =========================================================
def Axial_ROIs_estimation(
    slice_ROI_predicted,
    output_dir,
    table_dir="."
):
    

    # print("=" * 50)
    # print("Axial ROI detection + DICOM L/R correction + estimation")
    # print("=" * 50)

    offset_table, size_table = load_reference_tables(table_dir)

    timing = ROITiming("Axial")
    updated_slices = []
    estimated_coordinate_rows = []

    debug_dir = Path(output_dir) / "axial_estimation_debug"
    # --------------------------------------------------
    # Delete previous results
    # --------------------------------------------------
    if debug_dir.exists():
        shutil.rmtree(debug_dir)

    for slice_idx, slice_rois in enumerate(slice_ROI_predicted, start=1):

        # print(f"\nSlice index: {slice_idx}")
        # print("-" * 50)

        if len(slice_rois) == 0:
            print("No ROIs detected. Cannot estimate yet.")
            updated_slices.append(slice_rois)
            continue

        instance_number = slice_rois[0]["instance_number"]


        # -------------------------------------------------
        # Step 1: correct left/right labels using DICOM
        # -------------------------------------------------
        slice_rois = rename_left_right_regions_using_dicom(slice_rois)

        # print("\nAfter DICOM left/right correction:")

        for roi in slice_rois:
            old_name = roi.get("original_region_name", roi["region_name"])
            new_name = roi["region_name"]

            if old_name != new_name:
                print(
                    f"{old_name} -> {new_name}",
                    "conf:",
                    float(roi.get("confidence", 0.0))
                )
            else:
                print(
                    new_name,
                    "conf:",
                    float(roi.get("confidence", 0.0))
                )

        # -------------------------------------------------
        # Step 2: remove duplicates after correction
        # -------------------------------------------------
        original_count = len(slice_rois)

        slice_rois = keep_highest_confidence_per_region(slice_rois)

        removed_count = original_count - len(slice_rois)

        if removed_count > 0:
            print(f"\nRemoved duplicate ROIs: {removed_count}")
            print("Kept highest-confidence ROI for each corrected region.")

        detected_regions = {
            roi["region_name"]
            for roi in slice_rois
        }

        missing_regions = EXPECTED_REGIONS - detected_regions

        # print(f"\nDetected regions: {sorted(detected_regions)}")

        # -------------------------------------------------
        # Step 3: estimate missing ROIs
        # -------------------------------------------------
        if len(missing_regions) == 0:
            print("All 5 ROIs detected.")

            updated_slice_rois = sorted(
                slice_rois,
                key=lambda x: x["region"]
            )

        else:
            print(f"Missing regions: {sorted(missing_regions)}")

            dicom_slice = slice_rois[0]["dicom"]

            with timing.measure("ROI estimation", frame=instance_number):
                updated_slice_rois = estimate_missing_axial_rois(
                    dicom_slice=dicom_slice,
                    existing_rois=slice_rois,
                    missing_regions=missing_regions,
                    offset_table=offset_table,
                    size_table=size_table
                )

            # print("\nAfter estimation:")

            for roi in updated_slice_rois:
                status = (
                    "estimated"
                    if roi.get("estimated", False)
                    else "predicted"
                )

                print(
                    roi["region_name"],
                    roi["bbox"],
                    status
                )

                if roi.get("estimated", False):
                    print("  used refs:", roi["used_reference_regions"])
                    print("  weights:", roi["normalized_reference_weights"])

        # -------------------------------------------------
        # Step 4: save debug overlay and crops
        # -------------------------------------------------
        slice_path = save_axial_roi_debug_images(
            slice_rois=updated_slice_rois,
            save_dir=debug_dir
        )

        for roi in updated_slice_rois:
            roi["overlay_image_path"] = str(slice_path)
            dicom_obj = roi.get("dicom")
            if dicom_obj:
                roi["dicom_path"] = os.path.abspath(dicom_obj.filename) if getattr(dicom_obj, "filename", None) else ""

            if roi.get("estimated", False):
                estimated_coordinate_rows.append(
                    roi_to_coordinate_row(
                        plane="axial",
                        detector_type="Estimated",
                        roi=roi,
                    )
                )

        updated_slices.append(updated_slice_rois)

    timing.report(["ROI estimation"])
    append_coordinate_csv(
        axial_coordinates_csv_path(output_dir),
        estimated_coordinate_rows
    )



    return tuple(updated_slices)





























