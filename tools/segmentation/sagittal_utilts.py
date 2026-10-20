import os
from pathlib import Path
import io
from enum import Enum
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tools.segmentation.sagittal_segmentor import normalize_path

class SegmentationClass(Enum):
    Anterior = 0
    PosteriorA = 1
    PosteriorB = 2
    Vertebrae = 3
    IVD = 4
    Sacrum = 5
    
class LastIVDHolder:
    def __init__(self, value):
        self.value = value
    
def load_image(filename):
    mask =Image.open(filename)
    return mask

def image_to_bytes(image):
    # Save the image to a bytes buffer
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")  # Save as PNG (or the format of your choice)
    img_byte_arr.seek(0)  # Go to the start of the BytesIO buffer
    return img_byte_arr
    
def convertToCategoricalMasks(mask):
    # mask = np.asanyarray(mask)
    mask = np.copy(mask)
    unique_values = np.unique(mask)
    for i in range(len(unique_values)):
        mask[mask == unique_values[i]] = i
    return mask

def show_binary_mask(image):
    plt.figure()
    plt.imshow(image)
    plt.show()

def show_image(image, title="test"):
    plt.figure()
    # print(image.shape)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if len(image.shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.imshow(image)
        
    else:  # Grayscale image
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
        # print(image.shape)
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()




def getLast5IVDs(sagittal_mask):
    structure =  'IVD'
    structure_cls = SegmentationClass[structure].value
    mask = (sagittal_mask == structure_cls).astype(np.uint8) * 255
    ivd_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in ivd_contours]
    sorted_ivd_contours = sorted(zip(ivd_contours, bounding_boxes), key=lambda x: x[1][1], reverse=True) # sort from the lowest IVD to the highest
    sorted_ivd_contours = [item[0] for item in sorted_ivd_contours]  # Extract sorted contours only
    sorted_ivd_contours_filtered = [cnt for cnt in sorted_ivd_contours if len(cnt) > 75] # Post processing for wrong IVD class segmentation area, ***75*** is tuned by quick experiments
    last_5_ivd_contours = sorted_ivd_contours_filtered[:5]
    # print("\nlast_5_ivd_contours" ,len(last_5_ivd_contours))
    return last_5_ivd_contours 




def getIVDCentroid2D(last_5_ivd_contours):
    last_5_ivd_centriods = []
    for contour in last_5_ivd_contours:
        x, y, w, h = cv2.boundingRect(contour)  # Get bounding box
        cx, cy = x + w // 2, y + h // 2
        # print(f"Bounding Box Center: ({cx}, {cy})")
        last_5_ivd_centriods.append([cx,cy])
    return last_5_ivd_centriods


