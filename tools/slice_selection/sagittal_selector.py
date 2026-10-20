"""
Author: Osamah F. Abdulmahmod
Date: 2026

Description:
------------
This file implements sagittal slice selection using anatomical DICOM
patient coordinates instead of file order or InstanceNumber ordering.

The algorithm is robust to different scan acquisition directions
(left-to-right or right-to-left) because it uses the DICOM patient
coordinate system.

Main functionality:
-------------------
1. Compute the physical center location of each sagittal DICOM slice
   using:
       - ImagePositionPatient (IPP)
       - ImageOrientationPatient (IOP)
       - PixelSpacing

2. Determine anatomical LEFT and RIGHT sides using DICOM X coordinates:
       larger X  -> patient LEFT
       smaller X -> patient RIGHT

3. Sort sagittal slices anatomically:
       RIGHT --> MID --> LEFT

4. Compute the anatomical mid-plane:
       x_mid = (x_min + x_max) / 2

5. Select:
       - LEFT sagittal slices inside a distance range from mid-plane
       - RIGHT sagittal slices inside a distance range from mid-plane
       - One middle sagittal slice using:
             len(sorted_slices) // 2

Inputs:
-------
sagittal_dicoms:
    List of sagittal pydicom Dataset objects.

left_low_mm / left_high_mm:
    Distance threshold range for LEFT sagittal slices.

right_low_mm / right_high_mm:
    Distance threshold range for RIGHT sagittal slices.

Outputs:
--------
left_sagittal_dicoms:
    List of LEFT sagittal DICOM slices.

mid_sagittal_dicom:
    Middle sagittal DICOM slice.

right_sagittal_dicoms:
    List of RIGHT sagittal DICOM slices.

Optional:
---------
return_info=True:
    Returns additional metadata including:
        - InstanceNumbers
        - Anatomical side
        - Physical X coordinates
        - Distance from mid-plane
        - Number of selected slices

Applications:
-------------
- Foraminal stenosis slice localization
- Anatomical sagittal ROI selection
- Lumbar spine AI pipelines
- DICOM geometry-based slice analysis
"""

import numpy as np


def _get_slice_center_xyz(ds):
    """
    Compute real center point of the DICOM slice in patient coordinates.

    DICOM patient coordinate system:
        +X = patient LEFT
        -X = patient RIGHT
        +Y = patient POSTERIOR
        +Z = patient HEAD / SUPERIOR
    """

    ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
    iop = np.array(ds.ImageOrientationPatient, dtype=np.float64)

    row_dir = iop[:3]
    col_dir = iop[3:]

    pixel_spacing = np.array(ds.PixelSpacing, dtype=np.float64)

    row_spacing = pixel_spacing[0]
    col_spacing = pixel_spacing[1]

    rows = int(ds.Rows)
    cols = int(ds.Columns)

    center_xyz = (
        ipp
        + row_dir * col_spacing * ((cols - 1) / 2.0)
        + col_dir * row_spacing * ((rows - 1) / 2.0)
    )

    return center_xyz


def sagittal_slice_selection(
    sagittal_dicoms,
    left_low_mm,
    left_high_mm,
    right_low_mm,
    right_high_mm,
    return_info=False,
):
    """
    Return:
        left_sagittal_dicoms:
            list of LEFT sagittal DICOM slices inside [left_low_mm, left_high_mm]

        mid_sagittal_dicom:
            one middle sagittal DICOM slice using len(slices)//2

        right_sagittal_dicoms:
            list of RIGHT sagittal DICOM slices inside [right_low_mm, right_high_mm]
    """

    if sagittal_dicoms is None or len(sagittal_dicoms) == 0:
        raise ValueError("sagittal_dicoms is empty.")

    slice_rows = []

    for idx, ds in enumerate(sagittal_dicoms):

        required_tags = [
            "ImagePositionPatient",
            "ImageOrientationPatient",
            "PixelSpacing",
            "Rows",
            "Columns",
            "PixelData",
        ]

        for tag in required_tags:
            if not hasattr(ds, tag):
                raise ValueError(f"Missing required DICOM tag: {tag}")

        center_xyz = _get_slice_center_xyz(ds)

        x_mm = float(center_xyz[0])
        instance_number = int(ds.InstanceNumber) if hasattr(ds, "InstanceNumber") else idx

        slice_rows.append(
            {
                "index": idx,
                "dicom": ds,
                "x_mm": x_mm,
                "center_xyz": center_xyz,
                "instance_number": instance_number,
            }
        )

    # Anatomical sorting only:
    # smallest X = patient RIGHT
    # largest X  = patient LEFT
    slice_rows = sorted(slice_rows, key=lambda x: x["x_mm"])

    x_values = np.array([r["x_mm"] for r in slice_rows], dtype=np.float64)

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    x_mid = 0.5 * (x_min + x_max)

    for r in slice_rows:
        r["signed_dist_from_mid_mm"] = float(r["x_mm"] - x_mid)
        r["abs_dist_from_mid_mm"] = abs(r["signed_dist_from_mid_mm"])

        if r["x_mm"] < x_mid:
            r["patient_side"] = "RIGHT"
        elif r["x_mm"] > x_mid:
            r["patient_side"] = "LEFT"
        else:
            r["patient_side"] = "MID"

    # Middle slice by number of sagittal slices // 2
    mid_index = len(slice_rows) // 2
    mid_row = slice_rows[mid_index]
    mid_sagittal_dicom = mid_row["dicom"]

    # All RIGHT slices inside threshold band
    right_rows = [
        r for r in slice_rows
        if r["patient_side"] == "RIGHT"
        and right_low_mm <= r["abs_dist_from_mid_mm"] <= right_high_mm
    ]

    # All LEFT slices inside threshold band
    left_rows = [
        r for r in slice_rows
        if r["patient_side"] == "LEFT"
        and left_low_mm <= r["abs_dist_from_mid_mm"] <= left_high_mm
    ]

    if len(right_rows) == 0:
        raise ValueError(
            f"No RIGHT sagittal slices found inside range "
            f"[{right_low_mm}, {right_high_mm}] mm from mid-plane."
        )

    if len(left_rows) == 0:
        raise ValueError(
            f"No LEFT sagittal slices found inside range "
            f"[{left_low_mm}, {left_high_mm}] mm from mid-plane."
        )

    left_sagittal_dicoms = [r["dicom"] for r in left_rows]
    right_sagittal_dicoms = [r["dicom"] for r in right_rows]

    if return_info:
        info = {
            "rule": "DICOM patient coordinate: larger X = LEFT, smaller X = RIGHT",
            "x_min_right_mm": x_min,
            "x_max_left_mm": x_max,
            "x_mid_mm": x_mid,
            "num_total_sagittal_slices": len(slice_rows),
            "num_left_slices": len(left_sagittal_dicoms),
            "num_right_slices": len(right_sagittal_dicoms),
            "mid": {
                "side": "MID",
                "mid_index": mid_index,
                "instance_number": mid_row["instance_number"],
                "x_mm": mid_row["x_mm"],
                "distance_from_mid_mm": mid_row["abs_dist_from_mid_mm"],
            },
            "left": [
                {
                    "side": "LEFT",
                    "instance_number": r["instance_number"],
                    "x_mm": r["x_mm"],
                    "distance_from_mid_mm": r["abs_dist_from_mid_mm"],
                }
                for r in left_rows
            ],
            "right": [
                {
                    "side": "RIGHT",
                    "instance_number": r["instance_number"],
                    "x_mm": r["x_mm"],
                    "distance_from_mid_mm": r["abs_dist_from_mid_mm"],
                }
                for r in right_rows
            ],
        }

        return left_sagittal_dicoms, mid_sagittal_dicom, right_sagittal_dicoms, info

    return left_sagittal_dicoms, mid_sagittal_dicom, right_sagittal_dicoms