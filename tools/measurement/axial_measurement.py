import base64
import os
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from tools.measurement.sagittal_measurement import convert_to_builtin_types, uint16Touint8
from tools.segmentation.axial_segmenter import segment_axial_slice


class AxialSegmentationClass(Enum):
    IVD = 0
    PE = 1
    TS = 2
    AAP = 3
    BACKGROUND = 4


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


def encode_png_base64(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def get_image_segmentation_overlay(image, pred_mask, view="axial"):
    colors = np.array(
        [
            [255, 87, 51],
            [51, 255, 87],
            [51, 87, 255],
            [243, 255, 51],
            [255, 51, 243],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )

    overlay_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    unique_values = np.unique(pred_mask)

    background_class = pred_mask.max() if view == "axial" else pred_mask.min()

    for label in unique_values:
        label_int = int(label)
        if label_int == int(background_class):
            continue
        overlay_mask[pred_mask == label_int] = colors[label_int % len(colors)]

    if image.ndim == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_alpha = 0.25
    image_alpha = 0.75
    blended = image_rgb.copy()
    mask_pixels = np.any(overlay_mask > 0, axis=-1)
    blended[mask_pixels] = (
        image_rgb[mask_pixels] * image_alpha
        + overlay_mask[mask_pixels] * mask_alpha
    ).astype(np.uint8)
    return blended


def remap_aghm_axial_mask_to_measurement_mask(mask):
    """
    AGMH axial inference returns 0=background, 1=PE, 2=TS, 3=AAP, 4=IVD.
    The old axial measurement logic expects IVD=0, PE=1, TS=2, AAP=3,
    background=4.
    """
    mask = np.asarray(mask).astype(np.uint8)
    remapped = np.full(mask.shape, AxialSegmentationClass.BACKGROUND.value, dtype=np.uint8)
    remapped[mask == 1] = AxialSegmentationClass.PE.value
    remapped[mask == 2] = AxialSegmentationClass.TS.value
    remapped[mask == 3] = AxialSegmentationClass.AAP.value
    remapped[mask == 4] = AxialSegmentationClass.IVD.value
    return remapped


def convert_to_categorical_mask(mask):
    mask = np.copy(mask)
    unique_values = np.unique(mask)

    if len(unique_values) == 6:
        mask[mask == unique_values[0]] = unique_values[-1]
        unique_values = np.unique(mask)

    for idx, value in enumerate(unique_values):
        mask[mask == value] = idx

    return mask.astype(np.uint8)


def get_ivd_and_pe_class(mask):
    return np.where(
        (mask == AxialSegmentationClass.IVD.value)
        | (mask == AxialSegmentationClass.PE.value),
        mask,
        AxialSegmentationClass.BACKGROUND.value,
    ).astype(np.uint8)


def _largest_contour_for_label(segmentation_mask, label):
    binary = (segmentation_mask == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [contour for contour in contours if len(contour) > 100]

    if not contours:
        raise ValueError(f"No valid contour found for label {label}.")

    return max(contours, key=cv2.contourArea)


def get_contours(segmentation_mask):
    ivd_contour = _largest_contour_for_label(
        segmentation_mask,
        AxialSegmentationClass.IVD.value,
    )
    pe_contour = _largest_contour_for_label(
        segmentation_mask,
        AxialSegmentationClass.PE.value,
    )

    contours_with_boxes = [
        (ivd_contour, cv2.boundingRect(ivd_contour)),
        (pe_contour, cv2.boundingRect(pe_contour)),
    ]

    sorted_contours = sorted(contours_with_boxes, key=lambda x: x[1][1])
    return sorted_contours[0][0], sorted_contours[1][0]


def get_upper_convex_point(defects, pe_contour):
    max_y = float("inf")
    highest_defect = {}

    if defects is None:
        return highest_defect

    for idx in range(defects.shape[0]):
        s, e, f, d = defects[idx, 0]
        start = tuple(pe_contour[s][0])
        end = tuple(pe_contour[e][0])
        far = tuple(pe_contour[f][0])

        x1, y1 = start
        x2, y2 = end
        x3, y3 = far

        numerator = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
        denominator = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance = numerator / denominator if denominator != 0 else 0

        if distance >= 5 and y1 < max_y:
            max_y = y1
            highest_defect = {
                "start_line": start,
                "end_line": end,
                "convex_point": np.array(far),
                "distance": distance,
                "depth": d,
            }

    return highest_defect


def get_ivd_center(ivd_contour):
    moments = cv2.moments(ivd_contour)
    if moments["m00"] == 0:
        return (0, 0)
    return (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )


def get_convex_point_pe(pe_contour):
    pe_contour = np.array(pe_contour)
    hull = cv2.convexHull(pe_contour, returnPoints=False)
    defects = cv2.convexityDefects(pe_contour, hull)
    return get_upper_convex_point(defects, pe_contour)


def get_center_point_pe(pe_contour):
    pe_points = np.squeeze(np.array(pe_contour))
    x_min, x_max = np.min(pe_points[:, 0]), np.max(pe_points[:, 0])
    x_mid = (x_min + x_max) / 2
    distances = np.abs(pe_points[:, 0] - x_mid)
    closest_distance = np.min(distances)
    closest_x_points = pe_points[distances == closest_distance]
    return min(closest_x_points, key=lambda point: point[1])


def find_min_distance(upper_points, lower_points, pixel_spacing):
    min_distance = float("inf")
    upper_closest_point = None
    lower_closest_point = None
    spacing_x, spacing_y = pixel_spacing

    for pt1 in upper_points:
        for pt2 in lower_points:
            distance = np.linalg.norm(
                (np.array(pt1[0]) - np.array(pt2[0]))
                * np.array([spacing_x, spacing_y])
            )

            if distance < min_distance:
                min_distance = distance
                upper_closest_point = pt1[0]
                lower_closest_point = pt2[0]

    if upper_closest_point is None or lower_closest_point is None:
        raise ValueError("Unable to find foraminal landmark distance.")

    return min_distance, upper_closest_point, lower_closest_point


def get_foraminal_landmarks(ivd_contour, pe_contour, pixel_spacing):
    bottom_ivd_contour = [
        pt for pt in ivd_contour if pt[0][1] >= ivd_contour[:, 0, 1].mean()
    ]
    top_pe_contour = [
        pt for pt in pe_contour if pt[0][1] <= pe_contour[:, 0, 1].mean()
    ]

    ivd_middle_x = np.mean([pt[0][0] for pt in bottom_ivd_contour])
    pe_middle_x = np.mean([pt[0][0] for pt in top_pe_contour])

    left_bottom_ivd = [pt for pt in bottom_ivd_contour if pt[0][0] < ivd_middle_x]
    right_bottom_ivd = [pt for pt in bottom_ivd_contour if pt[0][0] >= ivd_middle_x]
    left_top_pe = [pt for pt in top_pe_contour if pt[0][0] < pe_middle_x]
    right_top_pe = [pt for pt in top_pe_contour if pt[0][0] >= pe_middle_x]

    left_dist, left_ivd_point, left_pe_point = find_min_distance(
        left_bottom_ivd,
        left_top_pe,
        pixel_spacing,
    )
    right_dist, right_ivd_point, right_pe_point = find_min_distance(
        right_bottom_ivd,
        right_top_pe,
        pixel_spacing,
    )

    return {
        "left_formenial_dist": left_dist,
        "left_formenial_IVD_point": left_ivd_point,
        "left_formenial_PE_point": left_pe_point,
        "right_formenial_dist": right_dist,
        "right_formenial_IVD_point": right_ivd_point,
        "right_formenial_PE_point": right_pe_point,
    }


def line_eq(left_point, right_point):
    r1, c1 = left_point
    r2, c2 = right_point
    dr = r2 - r1

    if dr == 0:
        a, b = 1, 0
    else:
        slope = (c2 - c1) / dr
        a, b = -slope, 1

    d = -(a * r1 + b * c1)
    return a, b, d


def get_q_point(ivd_contour, foraminal_landmarks, pe_convex_point):
    bottom_ivd_contour = [
        pt for pt in ivd_contour if pt[0][1] >= ivd_contour[:, 0, 1].mean()
    ]
    ivd_lower_points = np.array(bottom_ivd_contour).squeeze()
    left_ivd_point = foraminal_landmarks["left_formenial_IVD_point"]
    right_ivd_point = foraminal_landmarks["right_formenial_IVD_point"]

    a, b, d = line_eq(left_ivd_point, right_ivd_point)
    q_left_to_right_dist = np.abs(np.dot(ivd_lower_points, np.array([a, b])) + d)
    q_convex_dist = np.linalg.norm(pe_convex_point - ivd_lower_points, axis=1)
    q_idx = np.argmax(q_left_to_right_dist - q_convex_dist)

    return bottom_ivd_contour[q_idx][0]


def line_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        return None, x1

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def find_intersection(line1, line2):
    m1, c1 = line1
    m2, c2 = line2

    if m1 == m2:
        raise ValueError("Measurement lines are parallel.")

    if m1 is None:
        x = c1
        y = m2 * x + c2
    elif m2 is None:
        x = c2
        y = m1 * x + c1
    else:
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1

    return int(x), int(y)


def get_intersection_point(foraminal_landmarks, ivd_center_point, pe_convex_point):
    left_ivd_point = foraminal_landmarks["left_formenial_IVD_point"]
    right_ivd_point = foraminal_landmarks["right_formenial_IVD_point"]

    ivd_line = line_equation(tuple(left_ivd_point), tuple(right_ivd_point))
    center_line = line_equation(ivd_center_point, tuple(pe_convex_point))
    return find_intersection(ivd_line, center_line)


def detect_herniation(intersection_point, q_point, pe_convex_point, pixel_spacing):
    intersection_point = np.array(intersection_point)
    q_point = np.array(q_point)
    pe_convex_point = np.array(pe_convex_point)
    spacing_x, spacing_y = pixel_spacing

    dis_q = np.linalg.norm((pe_convex_point - q_point) * np.array([spacing_x, spacing_y]))
    dis_intersection = np.linalg.norm(
        (pe_convex_point - intersection_point) * np.array([spacing_x, spacing_y])
    )

    return dis_intersection >= dis_q, dis_q, dis_intersection


def get_q_projection_point(q_point, ivd_center, pe_convex_point):
    q_point, ivd_center, pe_convex_point = map(
        np.asarray,
        (q_point, ivd_center, pe_convex_point),
    )
    segment = pe_convex_point - ivd_center
    point_vector = q_point - ivd_center
    segment_norm_sq = np.dot(segment, segment)

    if segment_norm_sq == 0:
        return int(ivd_center[0]), int(ivd_center[1])

    t = np.dot(point_vector, segment) / segment_norm_sq
    projected = ivd_center + t * segment
    return int(projected[0]), int(projected[1])


def get_r_ratio(q_dash_point, pe_convex_point, dis_intersection, pixel_spacing):
    spacing_x, spacing_y = pixel_spacing
    if not dis_intersection:
        return float("inf")

    return (
        np.linalg.norm(
            (np.array(q_dash_point) - np.array(pe_convex_point))
            * np.array([spacing_x, spacing_y])
        )
        / dis_intersection
    )


def get_measurements_info(
    disc_herniation,
    r_ratio,
):
    return {
        "disc_herniation": bool(disc_herniation),
        "herniation_ratio": round(r_ratio, 3),
    }


def draw_text(info_text, overlay_image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 5, 12

    for idx, line in enumerate(info_text.split("\n")):
        cv2.putText(
            overlay_image,
            line,
            (x, y + idx * 12),
            font,
            0.35,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def draw_image(
    image,
    foraminal_landmarks,
    pe_convex_point,
    q_point,
    q_dash_point,
    ivd_center_point,
    intersection_point,
    radius=2,
):
    left_ivd_point = foraminal_landmarks["left_formenial_IVD_point"]
    left_pe_point = foraminal_landmarks["left_formenial_PE_point"]
    right_ivd_point = foraminal_landmarks["right_formenial_IVD_point"]
    right_pe_point = foraminal_landmarks["right_formenial_PE_point"]

    cv2.line(image, tuple(left_ivd_point), tuple(right_ivd_point), (255, 255, 0), radius)
    cv2.line(image, tuple(pe_convex_point), tuple(ivd_center_point), (255, 0, 255), radius)
    cv2.line(image, tuple(left_ivd_point), tuple(left_pe_point), (255, 140, 0), radius)
    cv2.line(image, tuple(right_ivd_point), tuple(right_pe_point), (255, 140, 0), radius)

    for point, color in [
        (pe_convex_point, (0, 255, 0)),
        (q_dash_point, (255, 0, 255)),
        (ivd_center_point, (0, 255, 255)),
        (intersection_point, (255, 255, 255)),
        (left_ivd_point, (255, 0, 0)),
        (left_pe_point, (255, 0, 0)),
        (right_ivd_point, (255, 0, 0)),
        (right_pe_point, (255, 0, 0)),
    ]:
        cv2.circle(image, tuple(point), radius, color, -1)


def applyAxialMeasurements(image, mask, pixel_spacing):
    if (mask > 100).any():
        mask_ivd_pe = convert_to_categorical_mask(mask)
    else:
        mask_ivd_pe = mask.astype(np.uint8)

    ivd_and_pe_mask = get_ivd_and_pe_class(mask_ivd_pe)
    ivd_contour, pe_contour = get_contours(ivd_and_pe_mask)

    try:
        pe_highest_defect = get_convex_point_pe(pe_contour)
        if pe_highest_defect:
            pe_convex_point = pe_highest_defect["convex_point"]
        else:
            pe_convex_point = get_center_point_pe(pe_contour)
    except Exception:
        pe_convex_point = get_center_point_pe(pe_contour)

    foraminal_landmarks = get_foraminal_landmarks(ivd_contour, pe_contour, pixel_spacing)
    q_point = get_q_point(ivd_contour, foraminal_landmarks, pe_convex_point)
    ivd_center_point = get_ivd_center(ivd_contour)
    intersection_point = get_intersection_point(
        foraminal_landmarks,
        ivd_center_point,
        pe_convex_point,
    )
    q_dash_point = get_q_projection_point(q_point, ivd_center_point, pe_convex_point)

    disc_herniation, dis_q, dis_intersection = detect_herniation(
        intersection_point,
        q_point,
        pe_convex_point,
        pixel_spacing,
    )
    r_ratio = get_r_ratio(q_dash_point, pe_convex_point, dis_intersection, pixel_spacing)

    info = get_measurements_info(
        disc_herniation,
        r_ratio,
    )

    if image.ndim == 2:
        measurements_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        measurements_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.drawContours(measurements_image, [ivd_contour], -1, (0, 255, 128), 2)
    cv2.drawContours(measurements_image, [pe_contour], -1, (51, 55, 255), 2)
    draw_image(
        measurements_image,
        foraminal_landmarks,
        pe_convex_point,
        q_point,
        q_dash_point,
        ivd_center_point,
        intersection_point,
    )

    info_text = (
        f"Disc Herniation: {info['disc_herniation']}\n"
        f"herniation_ratio: {info['herniation_ratio']}"
    )
    draw_text(info_text, measurements_image)

    return info, measurements_image


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


def _segmentation_results_by_level(axial_segmentation_results):
    if not axial_segmentation_results:
        return {}

    results = axial_segmentation_results.get("results", axial_segmentation_results)
    if isinstance(results, dict):
        results = list(results.values())

    mapped = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        if "level_name" in result:
            mapped[result["level_name"]] = result
        if "level_index" in result:
            mapped[int(result["level_index"])] = result

    return mapped


def processAxialMeasurements(
    axial_slice,
    device=None,
    MODEL_NAME="AGMH",
    patient_id=None,
    disc_level=None,
    pred_mask=None,
    mask_label_format="aghm",
    output_dir="output/measurement/axial",
):
    dicom_slice = _extract_dicom_from_slice_input(axial_slice)
    if dicom_slice is None:
        raise ValueError("No DICOM slice found for axial measurement.")

    output_dir = normalize_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pixel_spacing = [float(v) for v in getattr(dicom_slice, "PixelSpacing", [1.0, 1.0])]
    image = uint16Touint8(dicom_slice.pixel_array)

    if pred_mask is None:
        seg_result = segment_axial_slice(
            dicom_slice=dicom_slice,
            output_dir=output_dir / "segmentation",
            save_overlay=True,
            rotate_input_for_model=False,
            return_info=True,
            level_name=disc_level,
        )
        pred_mask = seg_result["mask"]
        mask_label_format = "aghm"
    else:
        seg_result = None

    raw_pred_mask = np.asarray(pred_mask).astype(np.uint8)

    if mask_label_format == "aghm":
        measurement_mask = remap_aghm_axial_mask_to_measurement_mask(raw_pred_mask)
    else:
        measurement_mask = raw_pred_mask

    safe_level = str(disc_level or "axial").replace("/", "-").replace("\\", "-")
    segmentation_overlay = get_image_segmentation_overlay(
        image,
        measurement_mask,
        view="axial",
    )

    try:
        axial_measurements_dict, axial_measurements_overlay_image = applyAxialMeasurements(
            image,
            measurement_mask,
            pixel_spacing=pixel_spacing,
        )
        axial_measurements_dict = convert_to_builtin_types(axial_measurements_dict)
        error = None
    except Exception as exc:
        axial_measurements_dict = {}
        axial_measurements_overlay_image = segmentation_overlay.copy()
        error = str(exc)

    measurements_overlay_path = output_dir / f"{safe_level}_herniation_overlay.png"
    cv2.imwrite(
        str(measurements_overlay_path),
        cv2.cvtColor(axial_measurements_overlay_image, cv2.COLOR_RGB2BGR),
    )

    return {
        "raw_pred_mask_unique_classes": np.unique(raw_pred_mask).tolist(),
        "measurement_mask_unique_classes": np.unique(measurement_mask).tolist(),
        "axial_measurements": {
            "measurements": axial_measurements_dict,
            "image_overlay_path": str(measurements_overlay_path),
            "error": error,
        },
        "pixel_spacing": pixel_spacing,
        "model_name": MODEL_NAME,
        "patient_id": patient_id,
        "disc_level": disc_level,
        "segmentation_result": seg_result,
    }


def applyAllDiscsAxialMeasurements(
    axial_required_slices_dict,
    device=None,
    MODEL_NAME="AGMH",
    patient_id=None,
    axial_segmentation_results=None,
    output_dir="output/measurement/axial",
    mask_label_format="aghm",
):
    diagnosis = {}
    seg_by_level = _segmentation_results_by_level(axial_segmentation_results)

    print("=" * 20, "axial", "=" * 20)

    for level_index, (disc_level, axial_slice) in enumerate(
        axial_required_slices_dict.items(),
        start=1,
    ):
        print(f"\nProcessing disc level: {disc_level}")
        herniation_key = f"{disc_level}_herniation"

        seg_result = seg_by_level.get(disc_level) or seg_by_level.get(level_index)
        pred_mask = None

        if isinstance(seg_result, dict) and "mask" in seg_result:
            pred_mask = seg_result["mask"]

        try:
            measurements_result = processAxialMeasurements(
                axial_slice=axial_slice,
                device=device,
                MODEL_NAME=MODEL_NAME,
                patient_id=patient_id,
                disc_level=disc_level,
                pred_mask=pred_mask,
                mask_label_format=mask_label_format,
                output_dir=output_dir,
            )
        except Exception as exc:
            diagnosis[herniation_key] = {
                "level": disc_level,
                "disc_herniation": None,
                "herniation_ratio": None,
                "image_overlay_path": None,
                "error": str(exc),
            }
            continue

        axial_measurements = measurements_result.get("axial_measurements", {})
        herniation_measurements = axial_measurements.get("measurements") or {}

        diagnosis[herniation_key] = {
            "level": disc_level,
            "disc_herniation": herniation_measurements.get("disc_herniation"),
            "herniation_ratio": herniation_measurements.get("herniation_ratio"),
            "image_overlay_path": axial_measurements.get("image_overlay_path"),
            "error": axial_measurements.get("error"),
        }

    return diagnosis
