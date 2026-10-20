import csv
from pathlib import Path


COORDINATE_CSV_COLUMNS = [
    "Instance_Number",
    "Detector_Type",
    "Region_Label",
    "Region_Name",
    "Confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "Plane",
    "Side",
    "Estimated",
    "Reference_Region",
    "Used_Reference_Regions",
    "Dicom_Path",
    "Overlay_Image_Path",
]


def _class_name(class_names, label):
    if class_names is None or label in ("", None):
        return ""

    try:
        return class_names.get(int(label), f"class_{label}")
    except Exception:
        return f"class_{label}"


def _confidence(value):
    if value in ("", None):
        return ""

    try:
        return float(value)
    except Exception:
        return value


def _box_values(box):
    if box is None:
        return ["", "", "", ""]

    try:
        values = list(box)
    except TypeError:
        return ["", "", "", ""]

    if len(values) != 4:
        return ["", "", "", ""]

    return [float(value) for value in values]


def _join_values(value):
    if value in ("", None):
        return ""

    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)

    return str(value)


def detection_to_coordinate_row(
    plane,
    detector_type,
    detection,
    class_names,
    instance_number,
    side="",
    dicom_path="",
    overlay_image_path="",
):
    label = detection.get("label", "")
    x1, y1, x2, y2 = _box_values(detection.get("box"))

    return {
        "Instance_Number": instance_number,
        "Detector_Type": detector_type,
        "Region_Label": label,
        "Region_Name": _class_name(class_names, label),
        "Confidence": _confidence(detection.get("score")),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "Plane": plane,
        "Side": side,
        "Estimated": False,
        "Reference_Region": "",
        "Used_Reference_Regions": "",
        "Dicom_Path": dicom_path,
        "Overlay_Image_Path": overlay_image_path,
    }


def roi_to_coordinate_row(plane, detector_type, roi, class_names=None):
    label = roi.get("region", roi.get("label", ""))
    region_name = (
        roi.get("region_name")
        or roi.get("level_name")
        or _class_name(class_names, label)
    )
    x1, y1, x2, y2 = _box_values(roi.get("bbox", roi.get("box")))

    return {
        "Instance_Number": roi.get("instance_number", ""),
        "Detector_Type": detector_type,
        "Region_Label": label,
        "Region_Name": region_name,
        "Confidence": _confidence(roi.get("confidence", roi.get("score"))),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "Plane": plane,
        "Side": roi.get("side", ""),
        "Estimated": bool(roi.get("estimated", False)),
        "Reference_Region": roi.get("reference_region", ""),
        "Used_Reference_Regions": _join_values(roi.get("used_reference_regions")),
        "Dicom_Path": roi.get("dicom_path", ""),
        "Overlay_Image_Path": roi.get("overlay_image_path", ""),
    }


def write_coordinate_csv(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COORDINATE_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def append_coordinate_csv(csv_path, rows):
    if not rows:
        return

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COORDINATE_CSV_COLUMNS)
        if should_write_header:
            writer.writeheader()
        writer.writerows(rows)
