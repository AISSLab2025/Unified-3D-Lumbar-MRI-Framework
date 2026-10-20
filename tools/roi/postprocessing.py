# import cv2
# import numpy as np


# def resize_keep_aspect(img, target_h=128, target_w=128):

#     h, w = img.shape[:2]

#     scale = min(target_w / float(w), target_h / float(h))

#     new_w = max(1, int(round(w * scale)))
#     new_h = max(1, int(round(h * scale)))

#     if scale < 1.0:
#         interp = cv2.INTER_AREA
#     else:
#         interp = cv2.INTER_LINEAR

#     resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

#     return resized


# def pad_to_fixed_size(img, final_h=128, final_w=128, pad_value=0):

#     h, w = img.shape[:2]

#     pad_top = (final_h - h) // 2
#     pad_bottom = final_h - h - pad_top

#     pad_left = (final_w - w) // 2
#     pad_right = final_w - w - pad_left

#     padded = cv2.copyMakeBorder(
#         img,
#         pad_top,
#         pad_bottom,
#         pad_left,
#         pad_right,
#         borderType=cv2.BORDER_CONSTANT,
#         value=pad_value
#     )

#     return padded


# def preprocess_roi_image(img, final_h=128, final_w=128):

#     img = resize_keep_aspect(
#         img,
#         target_h=final_h,
#         target_w=final_w
#     )

#     img = pad_to_fixed_size(
#         img,
#         final_h=final_h,
#         final_w=final_w,
#         pad_value=0
#     )

#     return img


# def post_processing_rois(rois, final_h=128, final_w=128):

#     processed_rois = []

#     for roi in rois:

#         pil_crop = roi["cropped_roi"]

#         if pil_crop is None:
#             continue

#         # PIL -> NumPy
#         np_crop = np.array(pil_crop)

#         # preprocess ROI
#         processed_np = preprocess_roi_image(
#             np_crop,
#             final_h=final_h,
#             final_w=final_w
#         )

#         # save processed ROI
#         new_roi = {

#             "level": roi["level"],

#             "bbox": roi["bbox"],

#             "confidence": roi["confidence"],

#             # original PIL image
#             "cropped_roi_pil": pil_crop,

#             # original numpy image
#             "cropped_roi_np": np_crop,

#             # processed image for classifier
#             "processed_roi": processed_np
#         }

#         processed_rois.append(new_roi)

#     return processed_rois



import cv2
import numpy as np


def resize_keep_aspect(img, target_h=128, target_w=128):
    h, w = img.shape[:2]

    scale = min(target_w / float(w), target_h / float(h))

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if scale < 1.0:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR

    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    return resized


def pad_to_fixed_size(img, final_h=128, final_w=128, pad_value=0):
    h, w = img.shape[:2]

    pad_top = (final_h - h) // 2
    pad_bottom = final_h - h - pad_top

    pad_left = (final_w - w) // 2
    pad_right = final_w - w - pad_left

    padded = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value
    )

    return padded


def preprocess_roi_image(img, final_h=128, final_w=128):
    img = resize_keep_aspect(
        img,
        target_h=final_h,
        target_w=final_w
    )

    img = pad_to_fixed_size(
        img,
        final_h=final_h,
        final_w=final_w,
        pad_value=0
    )

    return img


def post_processing_rois(rois, final_h=128, final_w=128):
    """
    Post-process ROI crops for classifier input while preserving
    all metadata returned by save_roi_results().

    Added fields:
        - cropped_roi_pil
        - cropped_roi_np
        - processed_roi

    Preserved fields:
        - level
        - bbox
        - cropped_roi
        - confidence
        - side
        - instance_number
        - level_name
        - overlay_image_path
        - any future fields added upstream
    """

    processed_rois = []

    for roi in rois:

        pil_crop = roi.get("cropped_roi")

        if pil_crop is None:
            continue

        # PIL -> NumPy
        np_crop = np.array(pil_crop)

        # Preprocess ROI for classifier
        processed_np = preprocess_roi_image(
            np_crop,
            final_h=final_h,
            final_w=final_w
        )

        # Preserve all existing ROI information
        new_roi = roi.copy()

        # Add additional representations
        # new_roi["cropped_roi_pil"] = pil_crop
        # new_roi["cropped_roi_np"] = np_crop
        new_roi["processed_roi"] = processed_np

        processed_rois.append(new_roi)

    return processed_rois