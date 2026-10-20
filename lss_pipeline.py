import os
import shutil
import tempfile
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any
from stage_timing import StageTiming

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "input"
OUTPUT_ROOT = PROJECT_ROOT / "output"
ULTRALYTICS_CONFIG_DIR = PROJECT_ROOT / ".ultralytics"

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import pydicom
from PIL import Image

from tools.classification.axial_central_classifier import Axial_central_classification
from tools.classification.axial_foraminal_classifier import Axial_foraminal_classification
from tools.classification.axial_lateral_classifier import Axial_lateral_classification
from tools.classification.axial_visualization_utils import save_combined_axial_slices
from tools.classification.sagittal_foraminal_classifier import Sagittal_ROI_classification
from tools.dicom.dicom_utils import getIVDsAxialSlices
from tools.dicom.volume_identifier import identifier_dataset
from tools.measurement.axial_measurement import applyAllDiscsAxialMeasurements
from tools.measurement.sagittal_measurement import getAllMeasurements
from tools.roi.axial_roi_detector import Axial_ROI_detection
from tools.roi.missing_roi_estimator import Axial_ROIs_estimation
from tools.roi.sagittal_roi_detector import Sagittal_ROI_detection
from tools.slice_selection.sagittal_selector import sagittal_slice_selection
from tools.segmentation.axial_segmenter import axial_segmentation
from tools.segmentation.sagittal_segmentor import sagittal_segmentation
from tools.segmentation.sagittal_utilts import getIVDCentroid2D, getLast5IVDs
from tools.measurement import sagittal_measurement

LEVEL_ORDER = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"]

SKIP_KEYS = {
    "cam",
    "cropped_roi",
    "dicom",
    "dicom_path",
    "expanded_crop",
    "image",
    "mask",
    "mask_224",
    "mask_224_channels",
    "mask_channels",
    "mask_model_space",
    "mask_model_space_channels",
    "pred_mask",
    "prob_224",
    "processed_roi",
    "segmentation_overlay",
    "segmentation_result",
}


def normalize_path(path_str: Any) -> Path:
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


def _safe_run_id(filename: str | None = None) -> str:
    stem = Path(filename or "study").stem
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return f"{safe_stem}_{uuid.uuid4().hex[:10]}"


def _clear_directory_contents(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    resolved_directory = directory.resolve()

    for item in directory.iterdir():
        resolved_item = item.resolve()
        if resolved_item == resolved_directory or resolved_directory not in resolved_item.parents:
            raise RuntimeError(f"Refusing to delete unsafe path: {resolved_item}")

        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def clear_previous_run_data() -> None:
    _clear_directory_contents(INPUT_ROOT)
    _clear_directory_contents(OUTPUT_ROOT)


def _is_dicom_file(path: Path) -> bool:
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True)
        return True
    except Exception:
        return False


def _infer_orientation(path: Path) -> str:
    lower_path = str(path).lower()
    if "sagittal" in lower_path:
        return "sagittal"
    if "axial" in lower_path:
        return "axial"

    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        iop = [float(x) for x in ds.ImageOrientationPatient]
        normal = np.cross(iop[:3], iop[3:])
        norm_val = np.linalg.norm(normal)
        if norm_val > 1e-6:
            normal = normal / norm_val
            if abs(normal[0]) > abs(normal[1]) and abs(normal[0]) > abs(normal[2]):
                return "sagittal"
            if abs(normal[2]) > abs(normal[0]) and abs(normal[2]) > abs(normal[1]):
                return "axial"
    except Exception:
        pass

    return "axial"


def extract_dicom_zip(zip_path: Path, run_id: str) -> Path:
    patient_dir = INPUT_ROOT
    axial_dir = patient_dir / "axial"
    sagittal_dir = patient_dir / "sagittal"
    axial_dir.mkdir(parents=True, exist_ok=True)
    sagittal_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        for root, _, files in os.walk(tmp_dir):
            for name in files:
                source = Path(root) / name
                if not source.is_file() or not _is_dicom_file(source):
                    continue

                target_dir = sagittal_dir if _infer_orientation(source) == "sagittal" else axial_dir
                target = target_dir / name
                if target.exists():
                    target = target_dir / f"{source.stem}_{uuid.uuid4().hex[:8]}{source.suffix}"
                shutil.copy2(source, target)

    return patient_dir


def _asset(path: Any, base_url: str | None) -> dict[str, str] | None:
    if not path:
        return None

    resolved = normalize_path(path)
    item = {"path": str(resolved)}
    try:
        relative = resolved.relative_to(OUTPUT_ROOT)
        if base_url:
            item["url"] = f"{base_url.rstrip('/')}/static/{relative.as_posix()}"
    except ValueError:
        pass
    return item


def _write_results_json(results: dict[str, Any], run_output: Path, base_url: str | None) -> dict[str, Any]:
    run_output.mkdir(parents=True, exist_ok=True)
    json_path = run_output / "results.json"
    clean_results = clean_for_json(results, base_url)
    _strip_roi_detection_urls(clean_results)
    clean_results["result_json"] = _asset(json_path, base_url)

    with json_path.open("w", encoding="utf-8") as f:
        import json

        json.dump(clean_results, f, indent=2, ensure_ascii=False)

    return clean_results


def _strip_asset_urls(obj: Any) -> None:
    if isinstance(obj, dict):
        if "path" in obj and "url" in obj and set(obj).issubset({"path", "url"}):
            obj.pop("url", None)
            return

        for value in obj.values():
            _strip_asset_urls(value)
    elif isinstance(obj, list):
        for item in obj:
            _strip_asset_urls(item)


def _strip_roi_detection_urls(results: dict[str, Any]) -> None:
    for plane in ("sagittal", "axial"):
        roi_detection = results.get(plane, {}).get("roi_detection")
        _strip_asset_urls(roi_detection)


def _clean_asset_dict(obj: dict[str, Any]) -> dict[str, str] | None:
    if "path" not in obj or not set(obj).issubset({"path", "url"}):
        return None

    path_value = obj.get("path")
    url_value = obj.get("url")

    if isinstance(path_value, dict):
        url_value = url_value or path_value.get("url")
        path_value = path_value.get("path")

    if path_value is None:
        return None

    asset = {"path": str(path_value)}
    if url_value:
        asset["url"] = str(url_value)
    return asset


def clean_for_json(obj: Any, base_url: str | None = None) -> Any:
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, Image.Image):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return _asset(obj, base_url)
    if isinstance(obj, str):
        normalized = normalize_path(obj)
        if normalized.exists() and OUTPUT_ROOT in normalized.parents:
            return _asset(normalized, base_url)
        return obj
    if isinstance(obj, dict):
        asset = _clean_asset_dict(obj)
        if asset is not None:
            return asset

        cleaned = {}
        for key, value in obj.items():
            if key in SKIP_KEYS:
                continue
            cleaned_value = clean_for_json(value, base_url)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value
        return cleaned
    if isinstance(obj, (list, tuple)):
        return [
            cleaned_item
            for item in obj
            if (cleaned_item := clean_for_json(item, base_url)) is not None
        ]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def _dicom_original_png(dicom_slice: Any, path: Path) -> str:
    arr = dicom_slice.pixel_array.astype(np.float32)
    arr -= np.min(arr)
    max_val = np.max(arr)
    if max_val > 0:
        arr /= max_val
    img = (arr * 255).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


def _result_asset_fields(result: dict[str, Any], base_url: str | None) -> dict[str, Any]:
    return {
        "original_image": _asset(result.get("original_image_path"), base_url),
        "overlay_image": _asset(result.get("overlay_path"), base_url),
        "unique_classes": result.get("unique_classes"),
        "instance_number": result.get("instance_number"),
    }


def _add_axial_classification_urls(
    classification_groups: list[list[dict[str, Any]]],
    combined_image_paths: dict[str, str],
    base_url: str | None,
) -> None:
    for group in classification_groups:
        for result in group:
            level_name = result.get("level_name")
            combined_path = combined_image_paths.get(level_name)
            combined_asset = _asset(combined_path, base_url)
            if combined_asset and combined_asset.get("url"):
                result["url"] = combined_asset["url"]


def run_lss_pipeline(patient_path: Path, run_id: str, base_url: str | None = None, timing=None) -> dict[str, Any]:
    timing = timing or StageTiming(run_id)
    patient_path = normalize_path(patient_path)
    run_output = OUTPUT_ROOT

    for folder in [
        run_output / "ROIs_detection" / "sagittal",
        run_output / "ROIs_detection" / "axial",
        run_output / "segmentation" / "sagittal",
        run_output / "segmentation" / "axial",
        run_output / "measurement" / "sagittal",
        run_output / "measurement" / "axial",
        run_output / "classification" / "sagittal",
        run_output / "classification" / "axial",
        run_output / "classification" / "axial_combined",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    with timing.measure("dicom_identification"):
        sagittal_volume, axial_volume, sagittal_dicoms, axial_dicoms = identifier_dataset(patient_path)

    left_sagittal_dicoms, mid_sagittal_dicom, right_sagittal_dicoms, sagittal_info = timing.call("sagittal_slice_selection", sagittal_slice_selection,
        sagittal_dicoms,
        left_low_mm=8.8,
        left_high_mm=24.8,
        right_low_mm=8.8,
        right_high_mm=24.8,
        return_info=True,
    )

    left_sagittal_rois, right_sagittal_rois = timing.call("sagittal_roi_detection", Sagittal_ROI_detection,
        left_sagittal_dicoms,
        right_sagittal_dicoms,
        output_dir=run_output,
        config_path=PROJECT_ROOT / "checkpoints" / "roi_detection" / "Sagittal_config.json",
    )

    with timing.measure("sagittal_roi_classification"):
        left_sagittal_cls = Sagittal_ROI_classification(
            left_sagittal_rois,
            output_dir=run_output / "classification" / "sagittal",
            config_path=PROJECT_ROOT / "checkpoints" / "classification" / "Sagittal_foramina_config.json",
        )
        right_sagittal_cls = Sagittal_ROI_classification(
            right_sagittal_rois,
            output_dir=run_output / "classification" / "sagittal",
            config_path=PROJECT_ROOT / "checkpoints" / "classification" / "Sagittal_foramina_config.json",
        )

    with timing.measure("sagittal_segmentation"):
        sagittal_seg = sagittal_segmentation(
            mid_sagittal_dicom,
            output_dir=run_output / "segmentation" / "sagittal",
            rotate_input_for_model=False,
            save_overlay=True,
            return_info=True,
        )
        sagittal_original = _dicom_original_png(
            mid_sagittal_dicom,
            run_output / "segmentation" / "sagittal" / "mid_sagittal_original.png",
        )

    with timing.measure("sagittal_measurements"):
        sagittal_measurement.set_measurement_output_dir(run_output / "measurement" / "sagittal")
        sagittal_diagnosis = getAllMeasurements(mid_sagittal_dicom, sagittal_seg["mask"], run_id)

    with timing.measure("axial_slice_selection"):
        last_5_ivd_contours = getLast5IVDs(sagittal_seg["mask"])
        last_5_ivd_centroids = getIVDCentroid2D(last_5_ivd_contours)
        projected = getIVDsAxialSlices(last_5_ivd_centroids, mid_sagittal_dicom, axial_dicoms)
        slice_number_list = [projected[level]["instance_number"] for level in LEVEL_ORDER]

    slice_1, slice_2, slice_3, slice_4, slice_5 = timing.call("axial_roi_detection", Axial_ROI_detection,
        axial_dicoms,
        slice_number_list=slice_number_list,
        output_dir=run_output,
        config_path=PROJECT_ROOT / "checkpoints" / "roi_detection" / "Axial_config.json",
    )

    slice_1, slice_2, slice_3, slice_4, slice_5 = timing.call("axial_roi_estimation", Axial_ROIs_estimation,
        slice_ROI_predicted=[slice_1, slice_2, slice_3, slice_4, slice_5],
        output_dir=run_output / "ROIs_detection" / "axial",
        table_dir=PROJECT_ROOT / "tools" / "roi",
    )

    axial_seg = timing.call("axial_segmentation", axial_segmentation,
        slice_1,
        slice_2,
        slice_3,
        slice_4,
        slice_5,
        output_dir=run_output / "segmentation" / "axial",
        save_overlay=True,
        rotate_input_for_model=False,
        return_info=True,
    )

    axial_required_slices = {
        "L1-L2": slice_1,
        "L2-L3": slice_2,
        "L3-L4": slice_3,
        "L4-L5": slice_4,
        "L5-S1": slice_5,
    }
    axial_measurements = timing.call("axial_measurements", applyAllDiscsAxialMeasurements,
        axial_required_slices,
        device=None,
        MODEL_NAME="AGMH",
        patient_id=run_id,
        axial_segmentation_results=axial_seg,
        output_dir=run_output / "measurement" / "axial",
    )

    with timing.measure("axial_roi_classification"):
        axial_foraminal_cls = Axial_foraminal_classification(
            slice_1, slice_2, slice_3, slice_4, slice_5,
            save_dir=run_output / "classification" / "axial",
            config_path=PROJECT_ROOT / "checkpoints" / "classification" / "Axial_foramina_config.json",
        )
        axial_lateral_cls = Axial_lateral_classification(
            slice_1, slice_2, slice_3, slice_4, slice_5,
            save_dir=run_output / "classification" / "axial",
            config_path=PROJECT_ROOT / "checkpoints" / "classification" / "Axial_lateral_config.json",
        )
        axial_central_cls = Axial_central_classification(
            slice_1, slice_2, slice_3, slice_4, slice_5,
            save_dir=run_output / "classification" / "axial",
            config_path=PROJECT_ROOT / "checkpoints" / "classification" / "Axial_central_config.json",
        )
        axial_combined_cls = save_combined_axial_slices(
            axial_foraminal_cls, axial_lateral_cls, axial_central_cls,
            save_dir=run_output / "classification" / "axial_combined",
        )
        _add_axial_classification_urls(
            [axial_foraminal_cls, axial_lateral_cls, axial_central_cls],
            axial_combined_cls, base_url,
        )

    timing.check_result("sagittal_measurements", sagittal_diagnosis)
    timing.check_result("axial_segmentation", axial_seg)
    timing.check_result("axial_measurements", axial_measurements)
    for stage, items in (
        ("sagittal_roi_classification", left_sagittal_cls),
        ("sagittal_roi_classification", right_sagittal_cls),
        ("axial_roi_classification", axial_foraminal_cls),
        ("axial_roi_classification", axial_lateral_cls),
        ("axial_roi_classification", axial_central_cls),
    ):
        timing.check_result(stage, items)
        if not items:
            timing.mark_partial(stage, "A classifier returned no ROI results.")

    results = {
        "status": "success",
        "run_id": run_id,
        "output_dir": str(run_output),
        "output_url": f"{base_url.rstrip('/')}/static" if base_url else None,
        "input": {
            "patient_path": str(patient_path),
            "sagittal_slices": len(sagittal_dicoms),
            "axial_slices": len(axial_dicoms),
            "sagittal_volume_shape": list(sagittal_volume.shape),
            "axial_volume_shape": list(axial_volume.shape),
        },
        "sagittal": {
            "slice_selection": sagittal_info,
            "roi_detection": {
                "left": left_sagittal_rois,
                "right": right_sagittal_rois,
            },
            "classification": {
                "left": left_sagittal_cls,
                "right": right_sagittal_cls,
            },
            "segmentation": {
                "original_image": _asset(sagittal_original, base_url),
                "overlay_image": _asset(sagittal_seg.get("overlay_path"), base_url),
                "unique_classes": sagittal_seg.get("unique_classes"),
            },
            "measurements": sagittal_diagnosis,
        },
        "axial": {
            "projected_slices": projected,
            "slice_number_list": slice_number_list,
            "roi_detection": {
                "L1-L2": slice_1,
                "L2-L3": slice_2,
                "L3-L4": slice_3,
                "L4-L5": slice_4,
                "L5-S1": slice_5,
            },
            "segmentation": {
                result["level_name"]: _result_asset_fields(result, base_url)
                for result in axial_seg["results"]
                if "level_name" in result and "error" not in result
            },
            "measurements": axial_measurements,
            "classification": {
                "foraminal": axial_foraminal_cls,
                "lateral": axial_lateral_cls,
                "central": axial_central_cls,
            },
        },
    }

    return results


def run_zip(zip_path: Path, filename: str | None = None, base_url: str | None = None) -> dict[str, Any]:
    run_id = _safe_run_id(filename)
    run_output = OUTPUT_ROOT
    timing = StageTiming(run_id, filename)

    try:
        with timing.measure("input_output_cleanup", overhead=True):
            clear_previous_run_data()
        with timing.measure("zip_extraction_and_dicom_routing", overhead=True):
            patient_path = extract_dicom_zip(zip_path, run_id)
        results = run_lss_pipeline(patient_path=patient_path, run_id=run_id, base_url=base_url, timing=timing)
    except Exception as exc:
        results = {
            "status": "failed",
            "run_id": run_id,
            "output_dir": str(run_output),
            "output_url": f"{base_url.rstrip('/')}/static" if base_url else None,
            "input_zip_filename": filename,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    with timing.measure("result_preparation", overhead=True):
        results = clean_for_json(results, base_url)
        _strip_roi_detection_urls(results)
        results["result_json"] = _asset(run_output / "results.json", base_url)
    results["timings"] = timing.save(run_output, PROJECT_ROOT / "benchmark", results["status"])
    # Write directly so null durations for stages that never ran remain explicit.
    import json
    with (run_output / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
    return results
