import os
from tools.roi.roi_timing import ROITiming
import sys
import json
import torch
import numpy as np

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from tools.roi.coordinate_csv import detection_to_coordinate_row, write_coordinate_csv



# ============================================================
# Model caches
# ============================================================
_YOLO_MODEL_CACHE = {}
_DINO_MODEL_CACHE = {}

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

def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)

    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clamp(min=0)

    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)

    union = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-7)




def ensemble_boxes_weighted(
    dino_boxes,
    dino_scores,
    dino_labels,
    yolo_boxes,
    yolo_scores,
    yolo_labels,
    iou_thr=0.4
):
    final_boxes = []
    final_scores = []
    final_labels = []

    unique_labels = set(dino_labels.tolist() + yolo_labels.tolist())

    for label in unique_labels:
        d_idx = (dino_labels == label).nonzero(as_tuple=True)[0]
        y_idx = (yolo_labels == label).nonzero(as_tuple=True)[0]

        d_b = dino_boxes[d_idx]
        d_s = dino_scores[d_idx]

        y_b = yolo_boxes[y_idx]
        y_s = yolo_scores[y_idx]

        used_d = [False] * len(d_b)
        used_y = [False] * len(y_b)

        if len(d_b) > 0 and len(y_b) > 0:
            ious = box_iou_xyxy(d_b, y_b)

            while True:
                max_iou = torch.max(ious)

                if max_iou < iou_thr:
                    break

                max_idx = torch.argmax(ious)

                d_i = max_idx // ious.shape[1]
                y_i = max_idx % ious.shape[1]

                score1 = float(d_s[d_i])
                score2 = float(y_s[y_i])

                new_box = (d_b[d_i] * score1 + y_b[y_i] * score2) / (score1 + score2)
                new_score = (score1 + score2) / 2.0

                final_boxes.append(new_box)
                final_scores.append(torch.tensor(new_score, device=dino_boxes.device))
                final_labels.append(label)

                used_d[d_i] = True
                used_y[y_i] = True

                ious[d_i, :] = -1.0
                ious[:, y_i] = -1.0

        for i in range(len(d_b)):
            if not used_d[i]:
                final_boxes.append(d_b[i])
                final_scores.append(d_s[i])
                final_labels.append(label)

        for i in range(len(y_b)):
            if not used_y[i]:
                final_boxes.append(y_b[i])
                final_scores.append(y_s[i])
                final_labels.append(label)

    if len(final_boxes) == 0:
        return (
            torch.zeros((0, 4), device=dino_boxes.device),
            torch.zeros((0,), device=dino_boxes.device),
            torch.zeros((0,), dtype=torch.int64, device=dino_boxes.device)
        )

    return (
        torch.stack(final_boxes),
        torch.stack(final_scores),
        torch.tensor(final_labels, dtype=torch.int64, device=dino_boxes.device)
    )


def normalize_path(path_str):
    if not path_str:
        return Path()
    path_str = str(path_str)
    if os.name == "nt" and path_str.startswith("/mnt/c/"):
        path_str = "C:/" + path_str[7:]
    elif os.name != "nt" and (path_str.startswith("C:/") or path_str.startswith("C:\\")):
        path_str = "/mnt/c/" + path_str[3:].replace("\\", "/")
    return Path(path_str)


class AxialROIDetector:

    _SINGLETON_INSTANCE = None

    def __init__(
        self,
        config_path="checkpoints/roi_detection/Axial_config.json"
    ):
        config_path = normalize_path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        yolo_ckpt_path = normalize_path(self.config["yolo_ckpt_path"])
        dino_ckpt_path = normalize_path(self.config["dino_ckpt_path"])
        dino_repo_path = normalize_path(self.config["dino_repo_path"])
        dino_config_path = self.config["dino_config_path"]

        num_classes = self.config["num_classes"]
        device = self.config.get("device", "auto")

        self.score_threshold = self.config.get("score_threshold", 0.2)
        self.ensemble_iou_threshold = self.config.get("ensemble_iou_threshold", 0.4)

        self.CLASS_NAMES = {
            int(k): v for k, v in self.config["class_names"].items()
        }

        self.COLOR_MAP = {
            int(k): tuple(v) for k, v in self.config["color_map"].items()
        }

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # ============================================================
        # Load YOLO
        # ============================================================
        yolo_key = f"{yolo_ckpt_path}:{self.device}"

        if yolo_key in _YOLO_MODEL_CACHE:
            self.yolo_model = _YOLO_MODEL_CACHE[yolo_key]
        else:
            if not Path(yolo_ckpt_path).exists():
                raise FileNotFoundError(f"YOLO checkpoint not found at: {yolo_ckpt_path}")

            yolo_model = YOLO(yolo_ckpt_path)
            _YOLO_MODEL_CACHE[yolo_key] = yolo_model
            self.yolo_model = yolo_model

        # ============================================================
        # Load DINO
        # ============================================================
        dino_repo = Path(dino_repo_path)

        if not dino_repo.exists():
            raise FileNotFoundError(f"DINO repository not found at: {dino_repo}")

        if str(dino_repo) not in sys.path:
            sys.path.insert(0, str(dino_repo))

        from util.slconfig import SLConfig
        from models import build_model

        cfg_path = dino_repo / dino_config_path

        if not cfg_path.exists():
            raise FileNotFoundError(f"DINO config file not found at: {cfg_path}")

        cfg = SLConfig.fromfile(str(cfg_path))
        cfg.device = self.device
        cfg.num_classes = num_classes

        if not hasattr(cfg, "backbone_dir"):
            cfg.backbone_dir = ""

        dino_key = f"{cfg_path}:{dino_ckpt_path}:{self.device}"

        if dino_key in _DINO_MODEL_CACHE:
            self.dino_model, self.postprocessors = _DINO_MODEL_CACHE[dino_key]
        else:
            self.dino_model, _, self.postprocessors = build_model(cfg)

            if not Path(dino_ckpt_path).exists():
                raise FileNotFoundError(f"DINO checkpoint not found at: {dino_ckpt_path}")

            ckpt = safe_torch_load(Path(dino_ckpt_path), map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

            self.dino_model.load_state_dict(state, strict=True)
            self.dino_model.to(self.device)
            self.dino_model.eval()

            _DINO_MODEL_CACHE[dino_key] = (self.dino_model, self.postprocessors)

    @classmethod
    def get_instance(cls, config_path=None):
        if cls._SINGLETON_INSTANCE is None:
            if config_path is None:
                cls._SINGLETON_INSTANCE = cls()
            else:
                cls._SINGLETON_INSTANCE = cls(config_path=config_path)

        return cls._SINGLETON_INSTANCE

    def _preprocess_dino(self, pil_img):
        img = np.array(pil_img).astype("float32") / 255.0

        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=-1)

        img_t = torch.from_numpy(img).permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        return (img_t - mean) / std

    def detect(self, image_input, return_all=False, timing=None):
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            raise TypeError("Unsupported image input type. Use PIL Image or numpy array.")

        timing = timing if timing is not None else ROITiming("Axial")
        w, h = pil_img.size

        # YOLO
        with timing.measure("YOLO detection", self.device):
            yolo_res = self.yolo_model.predict(
                source=pil_img,
                device=self.device,
                conf=self.score_threshold,
                verbose=False
            )[0]

            if yolo_res.boxes is not None and len(yolo_res.boxes) > 0:
                yolo_boxes = yolo_res.boxes.xyxy.to(self.device)
                yolo_scores = yolo_res.boxes.conf.to(self.device)
                yolo_labels = yolo_res.boxes.cls.to(torch.int64).to(self.device)
            else:
                yolo_boxes = torch.zeros((0, 4), device=self.device)
                yolo_scores = torch.zeros((0,), device=self.device)
                yolo_labels = torch.zeros((0,), dtype=torch.int64, device=self.device)

            # DINO
        with timing.measure("DINO detection", self.device):
            img_t = self._preprocess_dino(pil_img).to(self.device)

            with torch.no_grad():
                dino_outputs = self.dino_model([img_t])

            target_sizes = torch.tensor(
                [[h, w]],
                device=self.device,
                dtype=torch.float32
            )

            dino_processed = self.postprocessors["bbox"](dino_outputs, target_sizes)[0]

            dino_boxes = dino_processed["boxes"]
            dino_scores = dino_processed["scores"]
            dino_labels = dino_processed["labels"]

            keep = dino_scores >= self.score_threshold

            dino_boxes = dino_boxes[keep]
            dino_scores = dino_scores[keep]
            dino_labels = dino_labels[keep]

            # Ensemble
        with timing.measure("Ensemble merging", self.device):
            final_boxes, final_scores, final_labels = ensemble_boxes_weighted(
                dino_boxes,
                dino_scores,
                dino_labels,
                yolo_boxes,
                yolo_scores,
                yolo_labels,
                iou_thr=self.ensemble_iou_threshold
            )

        def format_boxes(boxes, scores, labels):
            results = []

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = [float(x) for x in box.cpu().tolist()]

                x1i = int(max(0, round(x1)))
                y1i = int(max(0, round(y1)))
                x2i = int(min(w, round(x2)))
                y2i = int(min(h, round(y2)))

                crop = pil_img.crop((x1i, y1i, x2i, y2i))

                results.append({
                    "box": [x1, y1, x2, y2],
                    "score": float(score.item()),
                    "label": int(label.item()),
                    "crop": crop
                })

            return results

        merged_results = format_boxes(final_boxes, final_scores, final_labels)

        if return_all:
            yolo_results = format_boxes(yolo_boxes, yolo_scores, yolo_labels)
            dino_results = format_boxes(dino_boxes, dino_scores, dino_labels)

            return merged_results, yolo_results, dino_results

        return merged_results


def save_roi_results(
    axial_dicoms,
    slice_number_list,
    output_dir,
    config_path
):
    vis_dir = f"{output_dir}/ROIs_detection/axial"
    # crops_dir = f"{output_dir}/axial/crops"

    os.makedirs(vis_dir, exist_ok=True)
    # os.makedirs(crops_dir, exist_ok=True)

    timing = ROITiming("Axial")
    detector = AxialROIDetector.get_instance(config_path=config_path)

    CLASS_NAMES = detector.CLASS_NAMES
    COLOR_MAP = detector.COLOR_MAP

    slice_number_list = [int(x) for x in slice_number_list]

    selected_dicoms = []

    for instance_number in slice_number_list:
        matched = None

        for ds in axial_dicoms:
            ds_instance = int(getattr(ds, "InstanceNumber", -9999))

            if ds_instance == instance_number:
                matched = ds
                break

        if matched is None:
            print(f"Warning: axial slice InstanceNumber={instance_number} not found.")
        else:
            selected_dicoms.append(matched)

    rois_by_instance = {int(x): [] for x in slice_number_list}
    coordinates_list = []

    def draw_and_save_predictions(pil_img, predictions, save_path):
        draw_img = pil_img.copy()
        draw = ImageDraw.Draw(draw_img)
        font = load_number_font()

        for det in predictions:
            box = det["box"]
            label = det["label"]

            region_name = CLASS_NAMES.get(label, f"class_{label}")
            number_text = AXIAL_COMBINED_ROI_NUMBERS.get(region_name, str(label + 1))
            color = AXIAL_ROI_DRAW_COLOR

            x1, y1, x2, y2 = box

            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
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
            draw.text((text_x, text_y), number_text, fill=color, font=font)

        draw_img.save(save_path)

    for ds in selected_dicoms:
        instance_num = int(getattr(ds, "InstanceNumber", -9999))

        try:
            arr = ds.pixel_array
        except Exception:
            print(f"Warning: Could not read pixel data from axial InstanceNumber={instance_num}.")
            continue

        if arr is None:
            print(f"Warning: Pixel array is None for axial InstanceNumber={instance_num}.")
            continue

        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        pil = Image.fromarray(arr).convert("RGB")

        merged_rois, yolo_rois, dino_rois = detector.detect(
            pil,
            return_all=True,
            timing=timing
        )

     
        merged_vis_path = f"{vis_dir}/merged_axial_slice_{instance_num}.png"

        draw_and_save_predictions(pil, merged_rois, merged_vis_path)

        # Sort ROIs by region label: LFS, LLS, SCS, RLS, RFS
        merged_rois = sorted(merged_rois, key=lambda x: x["label"])

        for roi_idx, r in enumerate(merged_rois):
            label = int(r["label"])
            region_name = CLASS_NAMES.get(label, f"class_{label}")

            roi_dict = {
                "region": label,
                "region_name": region_name,
                "bbox": r["box"],
                "cropped_roi": r["crop"],
                "confidence": r["score"],
                "instance_number": instance_num,
                "dicom": ds
            }

            rois_by_instance[instance_num].append(roi_dict)

            crop_img = r["crop"]

            if crop_img is not None:
                crop_filename = (
                    f"crop_axial_slice_{instance_num}_"
                    f"region_{region_name}_roi_{roi_idx}.png"
                )

                # crop_save_path = f"{crops_dir}/{crop_filename}"
                # crop_img.save(crop_save_path)

        for detector_name, rois in [
            ("YOLO", yolo_rois),
            ("DINO", dino_rois),
            ("Ensemble", merged_rois)
        ]:
            for det in rois:
                coordinates_list.append(
                    detection_to_coordinate_row(
                        plane="axial",
                        detector_type=detector_name,
                        detection=det,
                        class_names=CLASS_NAMES,
                        instance_number=instance_num,
                    )
                )

    timing.report(["YOLO detection", "DINO detection", "Ensemble merging"])

    csv_path = Path(output_dir) / "axial_coordinates.csv"
    write_coordinate_csv(csv_path, coordinates_list)

    print(f"Axial ROI detection outputs successfully saved to {output_dir}")

    return rois_by_instance


def Axial_ROI_detection( axial_dicoms,slice_number_list,  output_dir, config_path="checkpoints/roi_detection/Axial_config.json"):
    rois_by_instance = save_roi_results( axial_dicoms=axial_dicoms, slice_number_list=slice_number_list, output_dir=output_dir, config_path=config_path )

    ordered_slice_rois = []

    for instance_number in slice_number_list:
        instance_number = int(instance_number)
        ordered_slice_rois.append(rois_by_instance.get(instance_number, []))

    return tuple(ordered_slice_rois)
