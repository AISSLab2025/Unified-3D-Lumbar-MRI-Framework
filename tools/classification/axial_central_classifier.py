import os
import json
import csv
import argparse
import datetime
import time
from pathlib import Path
import multiprocessing as mp
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

_CENTRAL_CLASSIFICATION_DIR_CLEANED = False
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure robust imports across python execution contexts
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from tools.classification.models.axial_central_model import SwinWithAttention


def main():
    from utils import (
        set_seed,
        PrepOnlyDataset,
        HardFocalLoss,
        run_one_epoch,
        save_confusion_matrix,
        save_misclassifications_csv,
    )

    parser = argparse.ArgumentParser(description="Normal Testing Pipeline (No W&B)")
    default_config = Path(__file__).parent / "train_configuration.json"
    parser.add_argument("--config", type=str, default=str(default_config), help="Path to train_configuration.json used during training")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to the test dataset directory")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save test results")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Hyperparameters
    dropout_p = config.get("dropout_p", 0.45)
    focal_gamma = config.get("focal_gamma", 1.2)
    attention_mechanism = config.get("attention_mechanism", "none")
    attention_stage = config.get("attention_stage", "before_classification")
    cam_reduction = config.get("cam_reduction", 16)
    ugr_hidden_dim = config.get("ugr_hidden_dim", 256)
    ugr_dropout = config.get("ugr_dropout", 0.20)
    img_size = config.get("img_size", 128)
    num_workers = config.get("num_workers", 4)
    seed = config.get("seed", 42)
    
    num_classes = config.get("num_classes", 3)
    class_names = config.get("class_names", ["Normal", "Stenosis", "Severe_Stenosis"])
    label_map = config.get("label_map", {"0": 0, "1": 1, "2": 1, "3": 2})

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    test_dir = Path(args.test_dir) if args.test_dir else Path(config["test_dir"])
    model_path = Path(args.model_path)
    
    # Save test results inside the model's directory in a "testing" subfolder
    if args.save_dir:
        save_dir = Path(args.save_dir) / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        save_dir = model_path.parent / "testing" / f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    assert test_dir.exists(), f"Test directory does not exist: {test_dir}"
    assert model_path.exists(), f"Model file does not exist: {model_path}"

    print(f"Results will be saved to: {save_dir}")

    # =========================================================
    # TRANSFORMS & DATASET
    # =========================================================
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    test_dataset = PrepOnlyDataset(test_dir, label_map=label_map, class_names=class_names, transform=test_transform)
    
    print("Class names:", class_names)
    print("Test samples:", len(test_dataset))

    persistent_flag = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers > 0 else None

    test_loader_kwargs = dict(
        dataset=test_dataset,
        batch_size=128,  # Fixed batch size for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_flag
    )
    if prefetch_factor is not None:
        test_loader_kwargs["prefetch_factor"] = prefetch_factor

    test_loader = DataLoader(**test_loader_kwargs)

    # =========================================================
    # MODEL & LOSS
    # =========================================================
    model = SwinWithAttention(
        num_classes=num_classes,
        dropout_p=dropout_p,
        reduction=cam_reduction,
        attention_mechanism=attention_mechanism,
        attention_stage=attention_stage,
        ugr_hidden_dim=ugr_hidden_dim,
        ugr_dropout=ugr_dropout
    ).to(device)

    print(f"Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    print("Model loaded successfully.")

    # We use HardFocalLoss with alpha=None for standard evaluation
    criterion = HardFocalLoss(
        alpha=None,
        gamma=focal_gamma,
        reduction="mean"
    )

    # =========================================================
    # EVALUATION
    # =========================================================
    print("\nRunning evaluation...")
    test_loss, test_metrics, y_true, y_pred, y_prob, img_paths, old_labels, test_extra = run_one_epoch(
        model, test_loader, criterion, device, optimizer=None,
        aux_criterion=None, initial_loss_weight=0.0, use_aux_initial_loss=False
    )

    print("\n================ TEST RESULTS ================")
    print(f"Test Loss:           {test_loss:.4f}")
    print(f"Test Uncert:         {test_extra['mean_uncertainty']:.4f}")
    print(f"Test Accuracy:       {test_metrics['acc']:.4f}")
    print(f"Test PrecisionMacro: {test_metrics['precision_macro']:.4f}")
    print(f"Test SensitivityMacro:{test_metrics['recall_macro']:.4f}")
    print(f"Test F1Macro:        {test_metrics['f1_macro']:.4f}")
    print(f"Test PrecisionW:     {test_metrics['precision_weighted']:.4f}")
    print(f"Test SensitivityW:   {test_metrics['recall_weighted']:.4f}")
    print(f"Test F1W:            {test_metrics['f1_weighted']:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    # =========================================================
    # SPECIFICITY CALCULATION
    # =========================================================
    specificities = {}
    macro_specificity = 0.0
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities[class_names[i]] = specificity
        macro_specificity += specificity
    
    if num_classes > 0:
        macro_specificity /= num_classes

    print(f"Test SpecificityMacro: {macro_specificity:.4f}")
    print("Per-Class Specificity:")
    for cname, spec in specificities.items():
        print(f"  {cname:>15}: {spec:.4f}")

    # =========================================================
    # ROC CURVE PLOTTING
    # =========================================================
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    y_prob_np = np.array(y_prob)
    n_classes = y_true_bin.shape[1]
    
    if n_classes == 1 and num_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))
        n_classes = 2

    class_aucs = {}
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i in range(num_classes):
        if i < len(colors):
            color = colors[i]
        else:
            color = 'black'
            
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_np[:, i])
            roc_auc = auc(fpr, tpr)
            class_aucs[class_names[i]] = roc_auc
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} (area = {roc_auc:0.3f})')
        else:
            class_aucs[class_names[i]] = 0.0
            plt.plot([0, 1], [0, 1], color=color, lw=2, linestyle='--',
                     label=f'ROC curve of class {class_names[i]} (No positive samples)')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / "test_roc_curve.png")
    plt.close()

    try:
        auc_macro = roc_auc_score(y_true, y_prob_np, average='macro', multi_class='ovr')
        auc_weighted = roc_auc_score(y_true, y_prob_np, average='weighted', multi_class='ovr')
    except Exception as e:
        print("Warning: Could not compute multi-class AUC:", e)
        auc_macro = 0.0
        auc_weighted = 0.0

    print(f"Test AUC Macro:      {auc_macro:.4f}")
    print(f"Test AUC Weighted:   {auc_weighted:.4f}")
    
    print("\nPer-Class AUC:")
    for cname, cauc in class_aucs.items():
        print(f"  {cname:>15}: {cauc:.4f}")

    # =========================================================
    # SAVE OUTPUTS
    # =========================================================
    save_confusion_matrix(
        cm,
        class_names,
        save_dir / "test_confusion_matrix.png",
        title="Test Confusion Matrix"
    )

    with open(save_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write("=============== TEST RESULTS ===============\n")
        f.write(f"Test Loss:           {test_loss:.6f}\n")
        f.write(f"Test Accuracy:       {test_metrics['acc']:.6f}\n")
        f.write(f"Test PrecisionMacro: {test_metrics['precision_macro']:.6f}\n")
        f.write(f"Test RecallMacro:    {test_metrics['recall_macro']:.6f}\n")
        f.write(f"Test F1Macro:        {test_metrics['f1_macro']:.6f}\n")
        f.write(f"Test PrecisionW:     {test_metrics['precision_weighted']:.6f}\n")
        f.write(f"Test RecallW:        {test_metrics['recall_weighted']:.6f}\n")
        f.write(f"Test F1W:            {test_metrics['f1_weighted']:.6f}\n")
        f.write(f"Test SpecificityMacro: {macro_specificity:.6f}\n")
        f.write(f"Test Uncert:         {test_extra['mean_uncertainty']:.6f}\n")
        f.write(f"Test AUC Macro:      {auc_macro:.6f}\n")
        f.write(f"Test AUC Weighted:   {auc_weighted:.6f}\n")
        f.write("\nPer-Class AUC:\n")
        for cname, cauc in class_aucs.items():
            f.write(f"  {cname:>15}: {cauc:.6f}\n")
        f.write("\nPer-Class Specificity:\n")
        for cname, spec in specificities.items():
            f.write(f"  {cname:>15}: {spec:.6f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))

    # Save Misclassifications
    save_misclassifications_csv(
        save_dir / "misclassified_samples.csv",
        y_true,
        y_pred,
        y_prob,
        img_paths,
        old_labels
    )

    # Save Predictions
    pred_csv_path = save_dir / "test_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "old_label_folder",
            "true_label_merged",
            "pred_label_merged",
            "prob_class_0",
            "prob_class_1",
            "prob_class_2",
            "correct"
        ])
        for path, old_lbl, t, p, prob in zip(img_paths, old_labels, y_true, y_pred, y_prob):
            probs_to_write = [float(prob[i]) if i < len(prob) else 0.0 for i in range(3)]
            writer.writerow([
                path,
                old_lbl,
                t,
                p,
            ] + probs_to_write + [int(t == p)])

    summary = {
        "model_path": str(model_path),
        "test_dir": str(test_dir),
        "save_dir": str(save_dir),
        "img_size": img_size,
        "batch_size": 128,
        "num_classes": num_classes,
        "class_names": class_names,
        "test_loss": float(test_loss),
        "test_acc": float(test_metrics["acc"]),
        "test_precision_macro": float(test_metrics["precision_macro"]),
        "test_recall_macro": float(test_metrics["recall_macro"]),
        "test_specificity_macro": float(macro_specificity),
        "test_f1_macro": float(test_metrics["f1_macro"]),
        "test_auc_macro": float(auc_macro),
        "test_auc_weighted": float(auc_weighted),
        "test_mean_uncertainty": float(test_extra["mean_uncertainty"]),
        "class_aucs": {k: float(v) for k, v in class_aucs.items()},
        "class_specificities": {k: float(v) for k, v in specificities.items()}
    }
    
    with open(save_dir / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    # =========================================================
    # GRAD-CAM
    # =========================================================
    print("\nGenerating Grad-CAM examples...")
    try:
        from gradcam import save_gradcam_examples
        gradcam_save_dir = save_dir / "gradcam"
        save_gradcam_examples(
            model=model,
            dataset=test_dataset,
            device=device,
            save_dir=gradcam_save_dir,
            class_names=class_names,
            num_per_class=10
        )
    except Exception as e:
        print("Failed to generate Grad-CAM examples:", e)

    print("\nSaved files:")
    print(save_dir / "test_confusion_matrix.png")
    print(save_dir / "test_roc_curve.png")
    print(save_dir / "test_classification_report.txt")
    print(save_dir / "test_predictions.csv")
    print(save_dir / "misclassified_samples.csv")
    print(save_dir / "test_summary.json")
    print(save_dir / "gradcam")
    print("\nTesting Done.")


# =========================================================
# Path Normalization Helper for cross-platform
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
# Resize + Padding
# =========================================================
def resize_keep_aspect(img, target_h=128, target_w=128):
    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image shape: {img.shape}")

    scale = min(target_w / float(w), target_h / float(h))

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def pad_to_fixed_size(img, final_h=128, final_w=128, pad_value=0):
    h, w = img.shape[:2]

    if h > final_h or w > final_w:
        raise ValueError(
            f"Image larger than target after resize: {(h, w)} > {(final_h, final_w)}"
        )

    pad_top = (final_h - h) // 2
    pad_bottom = final_h - h - pad_top
    pad_left = (final_w - w) // 2
    pad_right = final_w - w - pad_left

    return cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )


def preprocess_roi(img, final_h=128, final_w=128):
    img = resize_keep_aspect(img, target_h=final_h, target_w=final_w)
    img = pad_to_fixed_size(img, final_h=final_h, final_w=final_w)
    return img


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
# Expand bbox + crop
# =========================================================
def expand_and_crop_roi(full_image, bbox, region_name):
    img_h, img_w = full_image.shape[:2]

    xmin, ymin, xmax, ymax = bbox

    xmin = int(round(xmin))
    ymin = int(round(ymin))
    xmax = int(round(xmax))
    ymax = int(round(ymax))

    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        return None, None

    if region_name in ["LFS", "RFS", "LLS", "RLS"]:
        expand_left = int(round(width * 0.49))
        expand_right = int(round(width * 0.49))
        expand_top = int(round(height * 0.50))
        expand_bottom = int(round(height * 0.50))
    else:  # e.g., SCS (Spinal Canal Stenosis)
        center_x = xmin + width / 2.0
        center_y = ymin + height / 2.0
        new_width = width * 3.0
        new_height = height * 1.2

        new_xmin = max(0, int(round(center_x - new_width / 2.0)))
        new_xmax = min(img_w, int(round(center_x + new_width / 2.0)))
        new_ymin = max(0, int(round(center_y - new_height / 2.0)))
        new_ymax = min(img_h, int(round(center_y + new_height / 2.0)))

        if new_xmax <= new_xmin or new_ymax <= new_ymin:
            return None, None

        roi_crop = full_image[new_ymin:new_ymax, new_xmin:new_xmax]

        if roi_crop.size == 0:
            return None, None

        return roi_crop, [new_xmin, new_ymin, new_xmax, new_ymax]

    new_xmin = max(0, xmin - expand_left)
    new_xmax = min(img_w, xmax + expand_right)

    new_ymin = max(0, ymin - expand_top)
    new_ymax = min(img_h, ymax + expand_bottom)

    if new_xmax <= new_xmin or new_ymax <= new_ymin:
        return None, None

    roi_crop = full_image[new_ymin:new_ymax, new_xmin:new_xmax]

    if roi_crop.size == 0:
        return None, None

    return roi_crop, [new_xmin, new_ymin, new_xmax, new_ymax]


# =========================================================
# Select best SCS only
# =========================================================
def get_best_scs(slice_rois):
    best_rois = {}

    for roi in slice_rois:
        region_name = roi["region_name"]

        if region_name not in ["SCS"]:
            continue

        if region_name not in best_rois:
            best_rois[region_name] = roi
        else:
            if roi["confidence"] > best_rois[region_name]["confidence"]:
                best_rois[region_name] = roi

    return best_rois


# =========================================================
# Preprocess axial central canal ROIs
# =========================================================
def preprocess_axial_central_rois(
    slice_1,
    slice_2,
    slice_3,
    slice_4,
    slice_5,
    final_h=128,
    final_w=128,
):
    all_slices = [slice_1, slice_2, slice_3, slice_4, slice_5]

    level_names = {
        1: "L1-L2",
        2: "L2-L3",
        3: "L3-L4",
        4: "L4-L5",
        5: "L5-S1",
    }

    processed_results = []

    for level_idx, slice_rois in enumerate(all_slices, start=1):
        print(f"\n========== Level {level_idx}: {level_names[level_idx]} ==========")

        best_rois = get_best_scs(slice_rois)

        for region_name in ["SCS"]:
            if region_name not in best_rois:
                print(f"[Missing] {region_name} not found in {level_names[level_idx]}")
                continue

            roi = best_rois[region_name]

            full_image = dicom_to_uint8(roi["dicom"])

            expanded_crop, expanded_bbox = expand_and_crop_roi(
                full_image,
                roi["bbox"],
                roi["region_name"],
            )

            if expanded_crop is None:
                print(f"[Invalid Crop] {region_name} at {level_names[level_idx]}")
                continue

            processed_roi = preprocess_roi(
                expanded_crop,
                final_h=final_h,
                final_w=final_w,
            )

            result = {
                "level_index": level_idx,
                "level_name": level_names[level_idx],
                "region_name": region_name,
                "original_bbox": roi["bbox"],
                "expanded_bbox": expanded_bbox,
                "confidence": roi["confidence"],
                "instance_number": roi["instance_number"],
                "processed_roi": processed_roi,
                "expanded_crop": expanded_crop,
                "dicom": roi["dicom"],
            }

            processed_results.append(result)

            print(
                region_name,
                "| original bbox:", roi["bbox"],
                "| expanded bbox:", expanded_bbox,
                "| processed shape:", processed_roi.shape,
                "| confidence:", roi["confidence"],
                "| instance:", roi["instance_number"],
            )

    return processed_results


# =========================================================
# Grad-CAM Class
# =========================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []

        self.handles.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self.handles.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        output_dict = self.model(input_tensor)
        output = output_dict["final_logits"]

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        score = output[:, class_idx]
        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Handle [B, H, W, C] layout from Swin Transformer (e.g. model.norm)
        if activations.ndim == 4 and activations.shape[-1] in [96, 192, 384, 768] and activations.shape[1] == activations.shape[2]:
            activations = activations.permute(0, 3, 1, 2)
            gradients = gradients.permute(0, 3, 1, 2)

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        pred = torch.argmax(output, dim=1).item()

        uncertainty = output_dict["uncertainty"]
        if uncertainty.ndim > 0:
            uncertainty = uncertainty[0].item()
        else:
            uncertainty = uncertainty.item()

        return cam, pred, probs, uncertainty


# =========================================================
# Grad-CAM Visualization Helpers
# =========================================================
def denormalize_image(img_tensor):
    """
    Convert normalized tensor back to image in [0, 1].
    Input tensor shape: [1, 3, H, W]
    """
    img = img_tensor.detach().cpu().clone()
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0, 1)

    img = img.squeeze(0)
    img = img.permute(1, 2, 0).numpy()

    return img


def overlay_cam_on_image(img_rgb, cam, alpha=0.40):
    heatmap = plt.get_cmap("jet")(cam)[..., :3]
    overlay = (1 - alpha) * img_rgb + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def save_inference_gradcam(
    img_rgb,
    cam,
    overlay,
    pred,
    probs,
    uncertainty,
    class_names,
    save_path,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_rgb)
    ax1.set_title("Input")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(cam, cmap="jet")
    ax2.set_title("Grad-CAM")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title(
        f"Overlay\nPred: {class_names[pred]}\n"
        f"Prob: {np.round(probs, 3).tolist()}\n"
        f"Unc: {uncertainty:.3f}"
    )
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Grad-CAM Recovery & Visualization Helpers
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
# Inference Classification Function
# =========================================================
def Axial_central_classification(
    slice_1,
    slice_2,
    slice_3,
    slice_4,
    slice_5,
    save_dir="output/classification",
    config_path="checkpoints/classification/Axial_central_config.json",
    device=None,
):
    """
    Performs inference-time classification on axial central canal ROIs.
    """

    # 1. Normalize and verify paths
    save_dir = normalize_path(save_dir)
    if save_dir.name == "test" or save_dir.name == "gradcam":
        found_output_dir = None
        for sl in [slice_1, slice_2, slice_3, slice_4, slice_5]:
            for r in sl:
                overlay_path_str = r.get("overlay_image_path")
                if overlay_path_str:
                    try:
                        overlay_path = Path(overlay_path_str)
                        p = overlay_path
                        for _ in range(10):
                            if p.name.lower() == "output":
                                found_output_dir = p / "classification"
                                break
                            p = p.parent
                        if found_output_dir:
                            break
                    except Exception:
                        pass
            if found_output_dir:
                break
        if found_output_dir is not None:
            save_dir = found_output_dir
        else:
            save_dir = Path("output/classification")

    save_dir = normalize_path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = normalize_path(config_path)

    # 2. Load configuration
    config = {}
    if resolved_config_path.exists():
        try:
            with open(resolved_config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(
                f"Warning: Failed to load config from {resolved_config_path}: {e}. "
                "Using default fallback config."
            )

    model_path_str = config.get(
        "model_path",
        "checkpoints/classification/Axial_central.pth",
    )
    num_classes = config.get("num_classes", 3)
    class_names = config.get(
        "class_names",
        ["Normal", "Stenosis", "Severe_Stenosis"],
    )
    img_size = config.get("img_size", 128)
    dropout_p = config.get("dropout_p", 0.4862)
    attention_mechanism = config.get("attention_mechanism", "spatial")
    attention_stage = config.get("attention_stage", "stage_3_4")
    ugr_hidden_dim = config.get("ugr_hidden_dim", 256)
    ugr_dropout = config.get("ugr_dropout", 0.20)

    # 3. Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # 4. Load model
    model = SwinWithAttention(
        num_classes=num_classes,
        dropout_p=dropout_p,
        attention_mechanism=attention_mechanism,
        attention_stage=attention_stage,
        reduction=16,
        ugr_hidden_dim=ugr_hidden_dim,
        ugr_dropout=ugr_dropout,
    ).to(device)

    resolved_model_path = normalize_path(model_path_str)
    print(f"Loading weights from: {resolved_model_path}")

    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Model weight path does not exist: {resolved_model_path}"
        )

    checkpoint = torch.load(resolved_model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print("Axial central canal classification model loaded successfully.")

    # --- DELETE PREVIOUS SAVED GRAD-CAM RESULTS IN output/classification ---
    global _CENTRAL_CLASSIFICATION_DIR_CLEANED
    if not _CENTRAL_CLASSIFICATION_DIR_CLEANED:
        if save_dir.exists():
            for item in save_dir.iterdir():
                if item.is_file() and "_central" in item.name:
                    try:
                        item.unlink()
                    except Exception as e:
                        print(f"Warning: Could not delete previous result {item}: {e}")
        _CENTRAL_CLASSIFICATION_DIR_CLEANED = True

    # 5. Preprocess ROIs
    processed_central_rois = preprocess_axial_central_rois(
        slice_1,
        slice_2,
        slice_3,
        slice_4,
        slice_5,
        final_h=img_size,
        final_w=img_size,
    )

    # 6. Transform
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

    # 7. Setup Grad-CAM once
    if getattr(model, "attention_block", None) is not None:
        target_layer = model.attention_block
    else:
        target_layer = model.norm

    gradcam = GradCAM(model, target_layer)

    # 8. Inference loop
    results = []
    total_classification_time = 0.0

    print("\n========== Running Inference AXIAL CENTRAL CANAL CLASSIFIER ==========")

    # Group the preprocessed ROIs by level_name
    from collections import defaultdict
    rois_by_level = defaultdict(list)
    for roi in processed_central_rois:
        rois_by_level[roi["level_name"]].append(roi)

    # Dictionaries to track base image slices and output paths
    slice_visualizations = {}

    try:
        for level_name, slice_rois in rois_by_level.items():
            safe_level_name = level_name.replace("/", "-")
            gradcam_path = save_dir / f"{safe_level_name}_merged.png"

            # Load clean slice image
            first_roi = slice_rois[0]
            slice_img = None
            if "dicom" in first_roi and first_roi["dicom"] is not None:
                try:
                    full_image = dicom_to_uint8(first_roi["dicom"])
                    slice_img = cv2.cvtColor(full_image, cv2.COLOR_GRAY2BGR)
                except Exception as e:
                    print(f"Warning: Failed to load DICOM from roi['dicom']: {e}")

            if slice_img is None:
                dicom_path_str = first_roi.get("dicom_path")
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
                overlay_image_path_str = first_roi.get("overlay_image_path")
                if overlay_image_path_str:
                    normalized_overlay_path = normalize_path(overlay_image_path_str)
                    if normalized_overlay_path.exists():
                        slice_img = cv2.imread(str(normalized_overlay_path))

            if slice_img is None:
                slice_img = np.zeros((512, 512, 3), dtype=np.uint8)

            h_img, w_img = slice_img.shape[:2]

            # Run inference and recover CAM for each ROI in this slice
            inference_rois = []
            for roi in slice_rois:
                crop = roi["processed_roi"]
                if crop.dtype != np.uint8:
                    crop = crop.astype(np.uint8)

                image = Image.fromarray(crop)
                image_tensor = transform(image).unsqueeze(0).to(device)

                # Grad-CAM + prediction
                start_cls = time.perf_counter()
                cam, pred_idx, probs, uncertainty = gradcam.generate(image_tensor)
                total_classification_time += time.perf_counter() - start_cls
                pred_prob = float(probs[pred_idx])

                inference_rois.append({
                    "roi": roi,
                    "cam": cam,
                    "pred_idx": pred_idx,
                    "probs": probs,
                    "pred_prob": pred_prob,
                    "uncertainty": uncertainty,
                })

            # Sort ROIs by region index or region name: SCS (region name SCS)
            def get_sort_key(item):
                rname = item["roi"]["region_name"]
                if rname == "SCS":
                    return 0
                return 1

            inference_rois = sorted(inference_rois, key=get_sort_key)

            # Draw everything on the slice
            legend_lines = []

            # PIL Image and Draw context
            slice_img_rgb = cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB)
            pil_slice = Image.fromarray(slice_img_rgb)
            draw = ImageDraw.Draw(pil_slice)

            # Load Fonts
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

            for idx, inf_roi in enumerate(inference_rois, start=1):
                roi = inf_roi["roi"]
                cam = inf_roi["cam"]
                pred_idx = inf_roi["pred_idx"]
                pred_prob = inf_roi["pred_prob"]
                probs = inf_roi["probs"]
                uncertainty = inf_roi["uncertainty"]
                region_name = roi["region_name"]

                # Recover original geometry
                cam_recovered = recover_cam_to_expanded_crop(cam, roi["expanded_crop"], roi["processed_roi"])

                # Slice to get original crop
                xmin, ymin, xmax, ymax = roi["original_bbox"]
                ex_xmin, ex_ymin, ex_xmax, ex_ymax = roi["expanded_bbox"]

                x1_crop = int(max(0, round(xmin - ex_xmin)))
                y1_crop = int(max(0, round(ymin - ex_ymin)))
                x2_crop = int(min(roi["expanded_crop"].shape[1], round(xmax - ex_xmin)))
                y2_crop = int(min(roi["expanded_crop"].shape[0], round(ymax - ex_ymin)))

                original_crop = roi["expanded_crop"][y1_crop:y2_crop, x1_crop:x2_crop]
                cam_original = cam_recovered[y1_crop:y2_crop, x1_crop:x2_crop]

                # Overlay recovered CAM on the original ROI crop
                overlay_roi = overlay_cam_on_roi(original_crop, cam_original)

                # Paste overlayed ROI back into original slice image
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

                # Class color selection (RGB)
                class_name = class_names[pred_idx]
                if class_name == "Normal":
                    color_rgb = (0, 255, 0)
                elif class_name == "Stenosis":
                    color_rgb = (255, 255, 0)
                elif class_name == "Severe_Stenosis":
                    color_rgb = (255, 0, 0)
                else:
                    color_rgb = (255, 255, 255)

                # Draw bounding box (thickness=1)
                draw.rectangle([x1i, y1i, x2i, y2i], outline=color_rgb, width=1)

                # Draw Number centered above the bounding box with black background
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

                # Append legend line info
                det_conf = roi.get("confidence", 0.0)
                class_label = class_name.replace('_', ' ')
                legend_text = f"{idx} - {region_name} ({det_conf:.2f}) {class_label} ({pred_prob:.2f})"
                legend_lines.append((legend_text, color_rgb))

                # Add results dictionary
                results.append(
                    {
                        "level_index": roi["level_index"],
                        "level_name": level_name,
                        "region_name": region_name,
                        "original_bbox": roi["original_bbox"],
                        "expanded_bbox": roi["expanded_bbox"],
                        "detection_confidence": roi["confidence"],
                        "instance_number": roi["instance_number"],

                        "predicted_index": pred_idx,
                        "predicted_class": class_names[pred_idx],
                        "predicted_probability": pred_prob,
                        "probabilities": {
                            class_names[i]: float(probs[i])
                            for i in range(num_classes)
                        },
                        "uncertainty": float(uncertainty),
                        "gradcam_path": str(gradcam_path),
                        "dicom": roi["dicom"],
                        "cam": cam,
                        "expanded_crop": roi["expanded_crop"],
                        "processed_roi": roi["processed_roi"],
                    }
                )

                print(
                    f"Level: {level_name} | Region: {region_name} "
                    f"-> Predicted: {class_names[pred_idx]} "
                    f"(Prob: {pred_prob:.4f}, Uncertainty: {uncertainty:.4f})"
                )

            # Draw level name in top-left corner with drop-shadow
            draw.text((21, 21), level_name, fill=(0, 0, 0), font=title_font)
            draw.text((20, 20), level_name, fill=(255, 255, 255), font=title_font)

            # Draw Legend lines below the title
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

            # Convert back to BGR and save
            slice_img = cv2.cvtColor(np.array(pil_slice), cv2.COLOR_RGB2BGR)
            slice_visualizations[safe_level_name] = slice_img

    finally:
        gradcam.remove_hooks()

    print(f"\n[TIMING] Axial Central Classification Time: {total_classification_time:.4f} seconds")

    # Save visualization slices (DISABLED - Only combined visualization is saved)
    # for safe_level_name, slice_img in slice_visualizations.items():
    #     if slice_img is not None:
    #         gradcam_path = save_dir / f"{safe_level_name}_central.png"
    #         cv2.imwrite(str(gradcam_path), slice_img)

    return results


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
