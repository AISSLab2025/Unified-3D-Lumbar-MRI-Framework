# import os
# from pathlib import Path
# import io
# from enum import Enum
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import pydicom
# from glob import glob



# def getSagittalTagsInfo(dcm_file):
#     pixel_spacing = dcm_file.PixelSpacing
#     top_sagittal_position = dcm_file.ImagePositionPatient
#     top_sagittal_orientation = dcm_file.ImageOrientationPatient
#     return pixel_spacing, top_sagittal_position, top_sagittal_orientation


# def project_to_3d(image_patient_position, x, y, pix_spacing, image_orientation):
#     """Project 2D coordinates to 3D space using Image Position and Orientation."""
#     sx, sy, sz = image_patient_position # Image position
#     o0, o1, o2, o3, o4, o5 = image_orientation  # Image orientation
#     delx, dely = pix_spacing[0], pix_spacing[1]  # Pixel spacing
#     # Projected 3D coordinates
#     xx = o0 * delx * x + o3 * dely * y + sx
#     yy = o1 * delx * x + o4 * dely * y + sy
#     zz = o2 * delx * x + o5 * dely * y + sz
#     return np.array([xx, yy, zz])



# def sagittalToAxialProjection3DApproach(axial_dicom_folder, sagittal_point_3D,  plane="axial", reverse_sort=True):
#     dicom_files = glob(os.path.join(axial_dicom_folder, "*.ima"))
#     dicom_files_sorted = sorted(dicom_files, key=lambda x: os.path.basename(x))
#     # print(dicom_files_sorted)
    
#     dicoms = [pydicom.dcmread(f) for f in dicom_files_sorted]
#     num_slices = len(dicoms)
#     plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
#     projection_dict = {}   #, 'dicom_position': None  
#     orientation = np.array([ dcm.ImageOrientationPatient  for dcm in dicoms])# Shape: (m, 6)
#     positions  = np.array([ dcm.ImagePositionPatient  for dcm in dicoms])  # Shape: (m, 3)
#     ox, oy = orientation[:, :3], orientation[:, 3:]  # Orientation vectors
#     oz = np.cross(ox, oy)  # Normal vectors to axial planes (m, 3) 
   
#     t = sagittal_point_3D[None, :] - positions[None, :, :]  #Shape: (1, m, 3)
#     dis = np.zeros(num_slices)
#     for i in range(1):
#         for j in range(num_slices):
#             dis[j] = np.abs(np.dot(t[i, j], oz[j]))

#     # print(dis)
#     closest_index = dis.argmin()
#     # print('dis.shape', dis.shape)
#     # print('type(dis)', type(dis))
    
#     min_dist = dis[closest_index]
    
#     insctances_number_list = [int(dcm.InstanceNumber) for dcm in dicoms]    

#     projection_dict['diffs'] = dis
#     projection_dict['lowest_dis'] = min_dist
#     projection_dict['instance_number']= int(insctances_number_list[closest_index])
#     projection_dict['slice_idx']= closest_index

#     return  projection_dict, min_dist, dis



# def getIVDsAxialSlices(last_5_ivd_centriods, sagittal_dcm, axial_dicom_folder):
#     projected_IVDs_slice_num = {}
#     pixel_spacing, top_sagittal_position, top_sagittal_orientation = getSagittalTagsInfo(sagittal_dcm)
#     for ivd_centroid, disc_name in zip(last_5_ivd_centriods, ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]):
#         x_point, y_point = ivd_centroid[0], ivd_centroid[1] # ALERT Swapped becuase the nature of cv2 corrdinates systmes
#         point_3D = project_to_3d(top_sagittal_position, x_point, y_point, pixel_spacing, top_sagittal_orientation)
#         projection_dict, min_dist, dis = sagittalToAxialProjection3DApproach(axial_dicom_folder, point_3D,  plane="axial", reverse_sort=True)
#         slice_idx =projection_dict['slice_idx']
#         if disc_name == "L5-S1" and slice_idx == (len(dis) -1 ) : ##### Curing L5 Baise
#             slice_idx = slice_idx -1 
#         projected_IVDs_slice_num[disc_name] = slice_idx

#     return projected_IVDs_slice_num



import numpy as np


def getSagittalTagsInfo(dcm_file):
    pixel_spacing = dcm_file.PixelSpacing
    sagittal_position = dcm_file.ImagePositionPatient
    sagittal_orientation = dcm_file.ImageOrientationPatient
    return pixel_spacing, sagittal_position, sagittal_orientation


def project_to_3d(image_patient_position, x, y, pix_spacing, image_orientation):
    sx, sy, sz = image_patient_position
    o0, o1, o2, o3, o4, o5 = image_orientation

    delx = float(pix_spacing[0])
    dely = float(pix_spacing[1])

    xx = o0 * delx * x + o3 * dely * y + sx
    yy = o1 * delx * x + o4 * dely * y + sy
    zz = o2 * delx * x + o5 * dely * y + sz

    return np.array([xx, yy, zz], dtype=float)


def sagittalToAxialProjection3DApproach(axial_dicoms, sagittal_point_3D):
    """
    axial_dicoms: list of pydicom Dataset
    sagittal_point_3D: 3D point projected from sagittal image
    """

    if axial_dicoms is None or len(axial_dicoms) == 0:
        raise ValueError("axial_dicoms is empty.")

    # Sort axial dicoms by InstanceNumber
    dicoms = sorted(
        axial_dicoms,
        key=lambda dcm: int(getattr(dcm, "InstanceNumber", 0))
    )

    num_slices = len(dicoms)

    orientation = np.array(
        [np.array(dcm.ImageOrientationPatient, dtype=float) for dcm in dicoms]
    )

    positions = np.array(
        [np.array(dcm.ImagePositionPatient, dtype=float) for dcm in dicoms]
    )

    ox = orientation[:, :3]
    oy = orientation[:, 3:]
    oz = np.cross(ox, oy)

    dis = np.zeros(num_slices, dtype=float)

    for j in range(num_slices):
        t = sagittal_point_3D - positions[j]
        dis[j] = abs(np.dot(t, oz[j]))

    closest_index = int(np.argmin(dis))
    min_dist = float(dis[closest_index])

    projection_dict = {
        "diffs": dis,
        "lowest_dis": min_dist,
        "instance_number": int(dicoms[closest_index].InstanceNumber),
        "slice_idx": closest_index,
        "dicom": dicoms[closest_index],
    }

    return projection_dict, min_dist, dis


# def getIVDsAxialSlices(last_5_ivd_centroids, sagittal_dcm, axial_dicoms):
#     """
#     last_5_ivd_centroids: list of 2D centroids from sagittal segmentation
#     sagittal_dcm: mid sagittal DICOM
#     axial_dicoms: list of axial DICOM datasets from identifier_dataset
#     """

#     projected_IVDs_slice_num = {}

#     pixel_spacing, sagittal_position, sagittal_orientation = getSagittalTagsInfo(
#         sagittal_dcm
#     )

#     disc_names = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]

#     for ivd_centroid, disc_name in zip(last_5_ivd_centroids, disc_names):

#         x_point = float(ivd_centroid[0])
#         y_point = float(ivd_centroid[1])

#         point_3D = project_to_3d(
#             sagittal_position,
#             x_point,
#             y_point,
#             pixel_spacing,
#             sagittal_orientation
#         )

#         projection_dict, min_dist, dis = sagittalToAxialProjection3DApproach(
#             axial_dicoms,
#             point_3D
#         )

#         slice_idx = projection_dict["slice_idx"]

#         # Fix L5-S1 if projected to last axial slice
#         if disc_name == "L5-S1" and slice_idx == len(dis) - 1:
#             slice_idx -= 1

#         projected_IVDs_slice_num[disc_name] = {
#             "slice_idx": slice_idx,
#             "instance_number": projection_dict["instance_number"],
#             "distance": min_dist,
#         }

#     return projected_IVDs_slice_num



import numpy as np


def get_axial_scan_direction(axial_dicoms):
    iop = np.array(axial_dicoms[0].ImageOrientationPatient, dtype=float)

    row = iop[:3]
    col = iop[3:]

    normal = np.cross(row, col)

    ipp_first = np.array(axial_dicoms[0].ImagePositionPatient, dtype=float)
    ipp_last = np.array(axial_dicoms[-1].ImagePositionPatient, dtype=float)

    movement = np.dot(ipp_last - ipp_first, normal)

    if movement > 0:
        return "FEET_TO_HEAD"
    else:
        return "HEAD_TO_FEET"


def getIVDsAxialSlices(last_5_ivd_centroids, sagittal_dcm, axial_dicoms):
    projected_IVDs_slice_num = {}

    pixel_spacing, sagittal_position, sagittal_orientation = getSagittalTagsInfo(
        sagittal_dcm
    )

    disc_names = ["L5-S1", "L4-L5", "L3-L4", "L2-L3", "L1-L2"]

    scan_direction = get_axial_scan_direction(axial_dicoms)
    print("Axial scan direction:", scan_direction)

    for ivd_centroid, disc_name in zip(last_5_ivd_centroids, disc_names):

        x_point = float(ivd_centroid[0])
        y_point = float(ivd_centroid[1])

        point_3D = project_to_3d(
            sagittal_position,
            x_point,
            y_point,
            pixel_spacing,
            sagittal_orientation
        )

        projection_dict, min_dist, dis = sagittalToAxialProjection3DApproach(
            axial_dicoms,
            point_3D
        )

        slice_idx = projection_dict["slice_idx"]

        # Fix L5-S1 if projected to last axial slice
        if disc_name == "L5-S1" and slice_idx == len(dis) - 1:
            slice_idx -= 1

        # Shift axial slice based on scan direction
        if scan_direction == "HEAD_TO_FEET":
            slice_idx = max(0, slice_idx - 1)

        elif scan_direction == "FEET_TO_HEAD":
            slice_idx = min(len(axial_dicoms) - 1, slice_idx + 1)

        instance_number = int(axial_dicoms[slice_idx].InstanceNumber)

        projected_IVDs_slice_num[disc_name] = {
            "slice_idx": slice_idx,
            "instance_number": instance_number,
            "distance": min_dist,
            "scan_direction": scan_direction,
        }

    return projected_IVDs_slice_num



