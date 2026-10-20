# tools/dicom/volume_identifier.py

from pathlib import Path
import numpy as np
import pydicom


def load_dicom_series(series_dir):
    """
    Load one DICOM series folder and return:
        volume: np.ndarray [num_slices, height, width]
        dicoms: list of pydicom Dataset objects
    """

    series_dir = Path(series_dir)

    dicoms = []

    for file_path in series_dir.rglob("*"):
        if file_path.is_file():
            try:
                ds = pydicom.dcmread(str(file_path), force=True)
                if hasattr(ds, "PixelData"):
                    dicoms.append(ds)
            except Exception:
                continue

    if len(dicoms) == 0:
        raise ValueError(f"No DICOM files found in: {series_dir}")

    def sort_key(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "SliceLocation"):
            return float(ds.SliceLocation)
        elif hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
        else:
            return 0

    dicoms = sorted(dicoms, key=sort_key)

    volume = np.stack([ds.pixel_array for ds in dicoms], axis=0)

    return volume, dicoms


def identifier_dataset(patient_path):
    """
    Input:
        patient_path:
            Path to one patient folder.

    Expected structure:
        patient_path/
        ├── Axial/
        └── Sagittal/

    Return:
        sagittal_volume:
            NumPy array with shape [num_slices, height, width]

        axial_volume:
            NumPy array with shape [num_slices, height, width]

        sagittal_dicoms:
            Original sagittal DICOM datasets with metadata preserved

        axial_dicoms:
            Original axial DICOM datasets with metadata preserved
    """

    patient_path = Path(patient_path)

    axial_dir = patient_path / "Axial"
    if not axial_dir.exists():
        axial_dir = patient_path / "axial"

    sagittal_dir = patient_path / "Sagittal"
    if not sagittal_dir.exists():
        sagittal_dir = patient_path / "sagittal"

    if not patient_path.exists():
        raise FileNotFoundError(f"Patient folder not found: {patient_path}")

    if not axial_dir.exists():
        raise FileNotFoundError(f"Axial folder not found (searched Axial and axial): {axial_dir}")

    if not sagittal_dir.exists():
        raise FileNotFoundError(f"Sagittal folder not found (searched Sagittal and sagittal): {sagittal_dir}")

    axial_volume, axial_dicoms = load_dicom_series(axial_dir)
    sagittal_volume, sagittal_dicoms = load_dicom_series(sagittal_dir)

    return sagittal_volume, axial_volume, sagittal_dicoms, axial_dicoms







# import os
# import shutil
# import zipfile
# from pathlib import Path

# def identifier_dataset(zip_path, extract_workspace="/tmp/lumbar_agent/extracted_volumes"):
#     """
#     Input:
#         zip_path:
#             Path to the patient's ZIP archive containing Axial and Sagittal DICOMs.
#         extract_workspace:
#             Temporary directory where the ZIP contents will be extracted.

#     Expected structure inside ZIP (handles double nesting):
#         Any sub-level containing an /Axial/ and /Sagittal/ directory.

#     Return:
#         sagittal_volume: NumPy array with shape [num_slices, height, width]
#         axial_volume: NumPy array with shape [num_slices, height, width]
#         sagittal_dicoms: Original sagittal DICOM datasets with metadata preserved
#         axial_dicoms: Original axial DICOM datasets with metadata preserved
#     """
#     zip_path = Path(zip_path)
#     if not zip_path.exists():
#         raise FileNotFoundError(f"Provided ZIP archive not found at: {zip_path}")

#     # Set up a clean, specific workspace for this file to avoid conflicts
#     session_extract_dir = Path(extract_workspace) / zip_path.stem
#     if session_extract_dir.exists():
#         shutil.rmtree(session_extract_dir)
#     session_extract_dir.mkdir(parents=True, exist_ok=True)

#     # 1. Unzip the file
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(session_extract_dir)

#     # 2. Dynamic Directory Discovery (Solves nesting or casing mismatches)
#     axial_dir = None
#     sagittal_dir = None

#     for root, dirs, _ in os.walk(str(session_extract_dir)):
#         for d in dirs:
#             if d.lower() == "axial":
#                 axial_dir = Path(root) / d
#             elif d.lower() == "sagittal":
#                 sagittal_dir = Path(root) / d

#     # 3. Validation Guardrails
#     if not axial_dir or not axial_dir.exists():
#         shutil.rmtree(session_extract_dir)  # Clean up before crashing
#         raise FileNotFoundError(
#             f"Validation Error: Could not locate an 'Axial' directory anywhere inside the provided ZIP file."
#         )

#     if not sagittal_dir or not sagittal_dir.exists():
#         shutil.rmtree(session_extract_dir)
#         raise FileNotFoundError(
#             f"Validation Error: Could not locate a 'Sagittal' directory anywhere inside the provided ZIP file."
#         )

#     try:
#         # 4. Load the DICOM arrays using your existing module function
#         axial_volume, axial_dicoms = load_dicom_series(axial_dir)
#         sagittal_volume, sagittal_dicoms = load_dicom_series(sagittal_dir)
#     finally:
#         # 5. Clean up disk space completely since files are safely held in memory variables
#         if session_extract_dir.exists():
#             shutil.rmtree(session_extract_dir)

#     return sagittal_volume, axial_volume, sagittal_dicoms, axial_dicoms
