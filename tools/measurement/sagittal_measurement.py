
import torch
import cv2
import base64
import matplotlib.pyplot as plt
import numpy as np
import  os
from skimage.morphology import skeletonize, thin, medial_axis
from scipy.interpolate import CubicSpline
from tools.segmentation.sagittal_utilts import SegmentationClass, LastIVDHolder 

MEASUREMENT_OUTPUT_DIR = None


def set_measurement_output_dir(output_dir):
    global MEASUREMENT_OUTPUT_DIR
    MEASUREMENT_OUTPUT_DIR = str(output_dir) if output_dir else None



def uint16Touint8(image):
    image = np.asarray(image)

    # If already uint8, return directly
    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)

    min_val = np.min(image)
    max_val = np.max(image)

    # avoid division by zero
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)

    scaled_image = (image - min_val) / (max_val - min_val) * 255.0
    return scaled_image.astype(np.uint8)


def getImageSegmentationOverlay(image, pred_mask, view):
    image_rgb = np.stack([image, image, image], axis=-1)  # Convert the grayscale image to RGB
    blended_image = image_rgb.copy()

    # Create a colormap with 6 distinct colors
    colormap = plt.get_cmap("tab20")
    colors = colormap(np.linspace(0, 1, 6))

    background_class = None
    if view == "axial":
        background_class = pred_mask.max()
    elif view == "sagittal":
        background_class = pred_mask.min()

    image_alpha = 0.75
    overlay_alpha = 0.25

    for class_value in np.unique(pred_mask):
        if class_value == background_class:
            continue

        mask_pixels = pred_mask == class_value
        color = (colors[int(class_value) % len(colors), :3] * 255).astype(np.float32)
        blended_image[mask_pixels] = (
            image_rgb[mask_pixels].astype(np.float32) * image_alpha
            + color * overlay_alpha
        ).astype(np.uint8)

    return blended_image



def display_image(image, title=None, cmap='gray', save_dir='output/measurement/sagittal', patient_id=None, level=None, function_name=None):
    """Helper function to save images with proper formatting
    Args:
        image: The image to display/save
        title: Title for the plot
        cmap: Colormap to use
        save_dir: Directory to save images
        patient_id: Patient identifier
        level: Anatomical level (e.g., 'L4-L5')
        function_name: Name of the calling function
    """
    plt.figure(figsize=(10, 10))
    if title:
        plt.title(title)
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    
    if MEASUREMENT_OUTPUT_DIR is not None:
        save_dir = MEASUREMENT_OUTPUT_DIR

    # Create base output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # If patient_id is provided, create and use patient-specific subfolder
    if patient_id:
        save_dir = os.path.join(save_dir, f"patient_{patient_id}")
        os.makedirs(save_dir, exist_ok=True)
    
    # Build filename components
    components = []
    if function_name:
        components.append(function_name)
    if level:
        components.append(f"level_{level}")
    if title:
        # Clean title and add to components
        clean_title = "".join(x for x in title if x.isalnum() or x in [' ', '-', '_']).rstrip()
        clean_title = clean_title.replace(' ', '_').lower()
        components.append(clean_title)
    
    # Join components with underscores
    filename = "_".join(components) if components else 'plot'
    
    # Save the plot
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory
    # print(f"Saved plot to: {save_path}")
    return save_path


def convertToCategoricalMasks(mask):
    # mask = np.asanyarray(mask)
    mask = np.copy(mask)
    unique_values = np.unique(mask)
    for i in range(len(unique_values)):
        mask[mask == unique_values[i]] = i
    return mask
 

def getIVDAndPEClass(temp_mask, structure_names = ['IVD', 'Vertebrae', 'Sacrum'], default_background_value=0):
    mask = temp_mask.copy()
    # Filter the mask
    structure_values = {cls.value for cls in SegmentationClass if cls.name in structure_names}
    filtered_mask = np.where(np.isin(mask, list(structure_values)), mask, default_background_value).astype(np.uint8)
    return filtered_mask





def filteringContours(contours, structure):
    """
    Filter out small contours based on their size.
    Args:
        contours: List of contours where each contour is a numpy array
    Returns:
        List of filtered contours
    """
    # Calculate mean and std of contour sizes
    contour_sizes = [len(contour) for contour in contours]
    mean_size = np.mean(contour_sizes)
    std_size = np.std(contour_sizes)
    
    # Filter contours that are within 1 standard deviation of the mean
    # This will keep contours that are not too small
    INITIAL_SIZE_THRESHOLD = mean_size // 2
    if structure == "Vertebrae":
        min_size = max(mean_size -  2* std_size, INITIAL_SIZE_THRESHOLD)
    elif structure == "IVD":
        min_size = max(mean_size -  2 *std_size, INITIAL_SIZE_THRESHOLD) 
    else:
        min_size = max(mean_size -   std_size, 0)

    # Filter contours using list comprehension
    filtered_contours = [contour for contour in contours if len(contour) >= min_size]
    return filtered_contours





def getContours(segmentation_mask, structure_names = ['IVD', 'Vertebrae',  'Sacrum']):
    contours_dict = {}
    for structure in structure_names:
        structure_cls = SegmentationClass[structure].value
        # print("segmentation_mask", segmentation_mask)
        mask = (segmentation_mask == structure_cls).astype(np.uint8) * 255
   
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            raise ValueError(f"No contours found for structure: {structure}")

        # Sort contours by the y-coordinate of their bounding rectangle
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        sorted_contours = sorted(zip(contours, bounding_boxes), key=lambda x: x[1][1], reverse=True) #y-coordinate of their bounding rectangle
        sorted_contours = [item[0] for item in sorted_contours]  # Extract sorted contours only
        filtered_sorted_contours = filteringContours(sorted_contours, structure)
   
        contours_dict[structure] = filtered_sorted_contours
   
    return contours_dict




def getSacrumCornerPoints(sacrum_Contours):
    # Initialize variables to store the top-left and top-right points
    top_left = sacrum_Contours[0]
    top_right = sacrum_Contours[0]

    for point in sacrum_Contours:
        x, y = point

        # Check for top-left point (smallest x and smallest y)
        if x < top_left[0] or (x == top_left[0] and y < top_left[1]):
            top_left = (x, y)

        # Check for top-right point (largest x and smallest y)
        if (x == top_right[0] and y < top_right[1]):
            top_right = (x, y)
    return top_left, top_right    # Filter the mask




def getLastIVDCornerPoints(IVD_Contour):
    # Initialize variables to store the top-left and top-right points
    top_left = IVD_Contour[0]
    top_right = IVD_Contour[0]
    bottom_left = IVD_Contour[0]
    bottm_right = IVD_Contour[0]
    
    for point in IVD_Contour:
        x, y = point
        # if (x <= top_left[0] and y <= top_left[1]):
        if (x <= top_left[0] ):
            top_left = (x, y)
        if (x >= top_right[0] and y <= top_right[1]):
            top_right = (x, y)
        # if (x <= bottom_left[0] and y >= bottom_left[1]):
        if (y >= bottom_left[1]):
            bottom_left = (x, y)
        # if (x >= bottm_right[0] and y >= bottm_right[1]):
        if (x >= bottm_right[0] ):
            bottm_right = (x, y)
    return top_left, top_right, bottm_right, bottom_left



def getExtremeCorners(image, mask_cls, structure,last_IVD, contour, bbox, skeletonization = False):
    x, y, w, h = bbox
    # Step 2: Assign initial raw corners from the bounding box
    raw_corners = [
        (x, y),               # Superior-left
        (x + w - 1, y),       # Superior-right
        (x + w - 1, y + h - 1), # Inferior-right
        (x, y + h - 1)        # Inferior-left
    ]
    
    if skeletonization:

        skeleton = skeletonize(mask_cls // 255).astype(np.uint8)  # Skeletonize
        skeleton_coords = np.column_stack(np.where(skeleton > 0))   # Get non-zero coordinates of the skeleton 
        # Swap the columns to convert (y, x) to (x, y)
        skeleton_coords_swapped = skeleton_coords[:, [1, 0]]
        
        corrected_corners = []
        if structure== 'Sacrum':
            top_left, top_right = getSacrumCornerPoints(skeleton_coords_swapped)
            raw_corners = [top_left, top_right]
            
        for raw_corner in raw_corners:
            # Compute Euclidean distance to all skeleton points
            distances =  np.linalg.norm(skeleton_coords_swapped - raw_corner, axis=1)
            min_idx = np.argmin(distances)  # Index of closest skeleton point
            corrected_corners.append(skeleton_coords_swapped[min_idx])  # Assign corrected corner point
        for point in skeleton_coords:
            cv2.circle(image, tuple(point[::-1]), 1, (255, 255, 255), -1)  # Blue for skeleton
    else:
        # Step 3: Flatten the contour array to get a list of contour points
        contour_points = contour.reshape(-1, 2)
    
        # Step 4: Initialize corrected corners list
        corrected_corners = []
    
        # Step 5: Correct each raw corner based on nearest neighbor in contour points
        if structure== 'Sacrum':
            top_left, top_right = getSacrumCornerPoints(contour_points)
            raw_corners = [top_left, top_right]

        elif structure =='IVD':
            if last_IVD.value:
                # print("hhhhhhhhhhhhhhhhh")
                top_left, top_right, bottom_left, bottm_right = getLastIVDCornerPoints(contour_points)
                raw_corners = [top_left, top_right, bottom_left, bottm_right]
                last_IVD.value = False
                
        for raw_corner in raw_corners:
            distances = np.linalg.norm(contour_points - raw_corner, axis=1)
            closest_point_idx = np.argmin(distances)
            corrected_corners.append(tuple(contour_points[closest_point_idx]))

    return corrected_corners, raw_corners



def align_refined_contour_using_pca(contour, image):
    # Convert contour to a (N,2) array of points
    # Compute the minimum area bounding box
    rect = cv2.minAreaRect(contour)
    contour_pts = contour.reshape(-1, 2).astype(np.float32)
    box = cv2.boxPoints(rect)  # Get the four corner points
    box = np.intp(box)  # Convert to integer coordinates
    contour_pts = np.intp(contour_pts)
    # Compute PCA on the refined bounding box
    mean, eigenvectors = cv2.PCACompute(box.astype(np.float32), mean=None)

    # Determine the principal axes
    major_axis = eigenvectors[0]  # First eigenvector (longest)
    minor_axis = eigenvectors[1]  # Second eigenvector (shorter)
    # Ensure major axis corresponds to width if necessary (swap axes if needed)
    if np.linalg.norm(major_axis) < np.linalg.norm(minor_axis):
        major_axis, minor_axis = minor_axis, major_axis
    # Compute rotation angle based on major axis
    angle = np.arctan2(major_axis[1], major_axis[0])  # Angle in radians
    angle_degrees = np.degrees(angle)  # Convert to degrees

    # print("angle_degrees", angle_degrees)
    margin  = 0
    margined_angle_degrees = angle_degrees - margin if angle_degrees > 0 else angle_degrees - margin
    invers_angle = -(margined_angle_degrees)  # Convert to degrees

    # Rotate points to align with PCA axes
    rotation_matrix = cv2.getRotationMatrix2D(tuple(mean[0]), -invers_angle, 1.0)
    aligned_box_pts = cv2.transform(np.expand_dims(box, axis=0), rotation_matrix)[0]
    aligned_contours_pts = cv2.transform(np.expand_dims(contour_pts, axis=0), rotation_matrix)[0]

    # Draw eigenvectors (arrows)
    center = tuple(map(int, mean[0]))  # Mean center point
    length = 50  # Length of the eigenvectors for visualization
    p1 = (int(center[0] + length * major_axis[0]), int(center[1] + length * major_axis[1]))  # Major axis end point
    p2 = (int(center[0] + length * minor_axis[0]), int(center[1] + length * minor_axis[1]))  # Minor axis end point



    # Return mean[0] as pca_center
    return aligned_contours_pts, invers_angle, rect, mean[0]




def find_local_minimum_x(x_indices, y_indices, search_slice, center_x, center_y, overlay_image):
    """
    Finds the X-coordinate corresponding to the first local minimum Y-value within the search range ΔC.
    """
    x_mu_candidates = np.intersect1d(search_slice, x_indices)

    def fallback_points():
        for x_val in sorted(np.unique(x_indices), key=lambda value: abs(value - center_x)):
            upper_and_lower = sorted(y_indices[x_indices == x_val])
            if len(upper_and_lower) >= 2:
                return np.array([x_val, upper_and_lower[0]]), np.array([x_val, upper_and_lower[-1]])
        return np.array([center_x, center_y]), np.array([center_x, center_y])


    if len(x_mu_candidates) > 0:
        upper_points = []
        lower_points = []
        for x_val in x_mu_candidates:
            # print("x_val", x_val)
            upper_and_lower = sorted(y_indices[x_indices == x_val])
            if( len(upper_and_lower) < 2):
                continue

            # print("upper_and_lower y_indices", upper_and_lower)
            upper_point = [x_val, upper_and_lower[0]]
            lower_point = [x_val, upper_and_lower[-1]]
            
            upper_points.append(upper_point)
            lower_points.append(lower_point)

        if not upper_points or not lower_points:
            return fallback_points()

        # # Find the point with the lowest Y value (local minimum)
        mu_points = [p for p in upper_points if p[1] == max(p[1] for p in upper_points)]
        ml_points = [p for p in lower_points if p[1] == min(p[1] for p in lower_points)]
      
        # Find the point nearest to the center point based on X coordinates only
        mu_nearest_point = min(mu_points, key=lambda p: abs(p[0] - center_x))
        ml_nearest_point = min(ml_points, key=lambda p: abs(p[0] - center_x))
  
        return np.array(mu_nearest_point), np.array(ml_nearest_point)  # X-value corresponding to this Y
    return fallback_points()  # Fallback if no valid points in ΔC





def getDeformitiesLandMarks(structure, contour, min_bbox, corrected_corners, overlay_image):
    """
    Detect six vertebral body landmarks (Au, Al, Pu, Pl, Mu, Ml) from a given vertebral contour.
    
    Args:
        structure (str): Name of the structure (e.g., "Vertebrae").
        contour (numpy array): Contour points obtained from cv2.findContours().
        bbox (tuple): Bounding box (x, y, w, h) of the contour.
    
    Returns:
        dict: Dictionary with six landmark coordinates (x, y) for Au, Al, Pu, Pl, Mu, Ml
    """
    
    (center_x, center_y), (width, height), angle = min_bbox
    # Convert to integer (since pixel coordinates must be integers)
    center_x, center_y, width, height = int(center_x), int(center_y), int(width), int(height)
    ymin = center_y - height // 2
    ymax = center_y + height // 2
    xmin = center_x - width // 2
    xmax = center_x + width // 2
    # Define search range ΔC around the center X
    search_range = int(0.15 * width)
    search_slice = np.clip(np.arange(center_x - search_range, center_x + search_range), xmin, xmax -1 )


    contour_points = contour
    x_indices = contour_points[:, 0]
    y_indices = contour_points[:, 1]

    # Filter points above and below center_y and sort by X values
    margen = 0
    # upper_points = contour_points[(y_indices > ymin) & (y_indices < ymin + margen)]
    upper_points = contour_points[(y_indices < center_y)]
    upper_points = upper_points[np.argsort(upper_points[:, 0])]


    lower_points = contour_points[y_indices >= center_y ]
    lower_points = lower_points[np.argsort(lower_points[:, 0])]
    # print(upper_points)
    # Identify key landmarks using sorted segments
    percent = 30/100
    
    segment_len = int(len(upper_points)*percent)
    x_au = np.max(upper_points[:segment_len, 0])  # Max X in first percent% of sorted upper points
    x_pu = np.min(upper_points[-segment_len:, 0])  # Max X in last percent% of sorted upper points
    x_al = np.max(lower_points[:segment_len, 0])  # Max X in first percent% of sorted lower points
    x_pl = np.min(lower_points[-segment_len:, 0])  # Max X in last percent% of sorted lower points

    # Find the first minimum X in the search range ΔC
    mu, ml = find_local_minimum_x(x_indices, y_indices, search_slice, center_x, center_y, overlay_image)

    # Get Y-coordinates from contour
    
    y_au = np.min(y_indices[x_indices == x_au])
    y_pu = np.min(y_indices[x_indices == x_pu])
  
    
    y_al = np.max(y_indices[x_indices == x_al])
    y_pl = np.max(y_indices[x_indices == x_pl])
    
    
    
    return {
        "center point": (center_x, center_y),
        "Au": (x_au, y_au), "Al": (x_al, y_al),
        "Pu": (x_pu, y_pu), "Pl": (x_pl, y_pl),
        "Mu": mu, "Ml": ml
    }


def inverse_rotate_points(points, center, angle):
    """
    Inverse-rotate a list of (x, y) points around a center by the given angle (in degrees).
    """
    # Prepare points as Nx2 numpy array
    pts = np.array(points, dtype=np.float32)
    if pts.ndim == 1:
        pts = np.expand_dims(pts, axis=0)
    pts = np.expand_dims(pts, axis=0)  # shape (1, N, 2)
    # Get rotation matrix for the inverse angle
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # angle is positive for inverse
    pts_inv = cv2.transform(pts, M)[0]  # shape (N, 2)
    return pts_inv.astype(int)



def drawDeformitesLandMarks_on_original(image, landmarks, center, angle):
    """
    Draws vertebral landmarks on the original image by inverse-rotating them from aligned space.
    Args:
        image: The original image to draw on.
        landmarks: Dictionary of landmark names to (x, y) in aligned space.
        center: Center of rotation (from PCA alignment).
        angle: The angle used for alignment (from PCA, invers_angle). Use -angle for inverse.
    """
    ############## Attention ########################
    # Exclude 'center point' from drawing
    names = [name for name in landmarks.keys() if name != 'center point']
    points = [landmarks[name] for name in names]
    ############## Attention ########################

    # Inverse-rotate points back to original space
    points_orig = inverse_rotate_points(points, center, angle)  # angle to invert
    # Draw points
    for x, y in points_orig:
        cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)  # Red
    # Create a mapping from name to point for easy access
    landmark_points = dict(zip(names, points_orig))
    # Draw lines between landmark pairs (as in drawDeformitesLines)
    pairs = [("Au", "Al"), ("Pu", "Pl"), ("Mu", "Ml")]
    for a, b in pairs:
        if a in landmark_points and b in landmark_points:
            pt1 = tuple(map(int, landmark_points[a]))
            pt2 = tuple(map(int, landmark_points[b]))
            cv2.line(image, pt1, pt2, (255, 51, 51), 1)  # Same color as original
    


def getAssessments(landmarks, pixel_spacing):
    """
    Computes vertebral deformity assessments using landmark distances.
    """
    spacing_x, spacing_y = pixel_spacing  # Unpack pixel spacing

    # Compute distances using correct spacing
    def scaled_distance(point1, point2):
        return np.linalg.norm([
            (point1[0] - point2[0]) * spacing_x,  # X-coordinates (width)
            (point1[1] - point2[1]) * spacing_y   # Y-coordinates (height)
        ])

    # Compute vertebral heights in real-world units
    Ah = scaled_distance(landmarks["Au"], landmarks["Al"])
    Ph = scaled_distance(landmarks["Pu"], landmarks["Pl"])
    Mh = scaled_distance(landmarks["Mu"], landmarks["Ml"])

    # Compute deformity ratios
    wedge = 100 - (Ah / Ph) * 100 if Ah < Ph else 100 - (Ph / Ah) * 100
    biconcave = 100 - (Mh / Ph) * 100 if Ah < Ph else 100 - (Mh / Ah) * 100

    return {
        "Anterior Height": Ah, 
        "Posterior Height": Ph, 
        "Middle Height": Mh, 
        "Wedge Deformity": np.round(wedge, 3), 
        "Biconcave Deformity": np.round(biconcave, 3)
    }


def applyGenantGrading(assessments):
    """
    Applies Genant vertebral fracture grading based on height ratios for both wedge and biconcave deformities.
    
    Genant Classification:
    - Normal: <20% reduction in height
    - Mild: 20-25% reduction in height
    - Moderate: 25-40% reduction in height
    - Severe: >40% reduction in height
    
    Args:
        assessments (dict): Dictionary containing height measurements and deformity ratios
        
    Returns:
        dict: Updated assessments dictionary with Genant classifications for both deformities
    """
    # Get the deformity ratios
    wedge = assessments["Wedge Deformity"]
    biconcave = assessments["Biconcave Deformity"]
    
    # Function to determine classification based on deformity percentage
    def get_classification(deformity):
        if deformity < 20:
            return "Normal"
        elif 20 <= deformity < 25:
            return "Mild"
        elif 25 <= deformity < 40:
            return "Moderate"
        else:
            return "Severe"
    
    # Get classifications for both deformities
    wedge_classification = get_classification(wedge)
    biconcave_classification = get_classification(biconcave)
    
    # Add Genant classifications to the assessments
    assessments["Wedge Classification"] = wedge_classification
    assessments["Biconcave Classification"] = biconcave_classification
    
    return assessments



def applyVertebraeDeformitiesAssessmentsOnImage(image, mask, pixel_spacing, patient_id, required_structure_measurements= "Vertebrae", 
                                                contour_colors = None, skeletonization= False , debug_images="output/measurement/sagittal"):
    if MEASUREMENT_OUTPUT_DIR is not None:
        debug_images = MEASUREMENT_OUTPUT_DIR

    lumbar_verteberaes = ['S1','L5','L4','L3','L2','L1']
    strucure_labels = {
            'Vertebrae': ['L5','L4','L3','L2','L1','T12','T11','T10'],
        }
    contour_colors = {
        'IVD': (255, 0, 0),
        'Vertebrae': (0, 255, 0),
        'Sacrum': (0, 0, 255)
    }
    deformities_dict = {   }

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    text_color = (17, 255, 0)  # White

    overlay_image= cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=[ "Vertebrae"])  # Example structure names

    # Clean API build saves only user-facing measurement overlays.

    # Convert mask to 8-bit unsigned integer
    # mask_8u = (mask * 255).astype(np.uint8)
    # mask_vis = cv2.cvtColor(mask_8u, cv2.COLOR_GRAY2RGB)

    vertebrae_mask = (filtered_mask == SegmentationClass["Vertebrae"].value).astype(np.uint8) * 255
    vertebrae_mask_vis = cv2.cvtColor(vertebrae_mask, cv2.COLOR_GRAY2RGB)

    # Create and save segmentation overlay
    overlay = image.copy()
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

    # Create colored overlay for each class
    for structure in ['Vertebrae', 'IVD', 'Sacrum']:
        structure_cls = SegmentationClass[structure].value
        structure_mask = (mask == structure_cls).astype(np.uint8) * 255
        color = contour_colors.get(structure, (255, 62, 23))
        
        mask_pixels = structure_mask > 0
        if np.any(mask_pixels):
            color_array = np.asarray(color, dtype=np.float32)
            overlay[mask_pixels] = (
                overlay[mask_pixels].astype(np.float32) * 0.75
                + color_array * 0.25
            ).astype(np.uint8)
    
    # Draw each structure's contours
    for structure, sorted_contours in contours_dict.items():
        if structure == 'IVD':
            last_IVD = LastIVDHolder(True)
        else:
            last_IVD = LastIVDHolder(False)

        if structure =="Vertebrae":
            last_vertebrae = len(sorted_contours)
            # print(f"Number of vertebrae contours found: {last_vertebrae}")
        
        color = contour_colors.get(structure, (255, 62, 23))  # Default to white
        structure_cls = SegmentationClass[structure].value
        mask_cls = (filtered_mask == structure_cls).astype(np.uint8) * 255
        
        # Create a copy of the image for contour visualization
        contour_image = overlay_image.copy()
        
        for contour, label in zip(sorted_contours[:], strucure_labels[structure]):
            # Draw the original contour
            cv2.drawContours(contour_image, [contour], -1, color, 2)
            

            # Get bounding box and corners
            bbox = cv2.boundingRect(contour)
            corrected_corners, raw_corners = getExtremeCorners(overlay_image, mask_cls, structure, last_IVD, contour, bbox, skeletonization)
            
            # Align the contour
            aligned_contour, angle, min_bbox, pca_center = align_refined_contour_using_pca(contour, contour_image)
            
           
            # Get landmarks
            land_marks = getDeformitiesLandMarks(structure, aligned_contour, min_bbox, corrected_corners, contour_image)
            
            ################## Attention: If you want to draw the rotated Contoures #####################################
            # Draw landmarks on aligned (rotated) contour
            # drawDeformitesLandMarks(contour_image, land_marks)
            # drawDeformitesLines(contour_image, land_marks)
            ################## Attention: If you want to draw the rotated Contoures #####################################
            
            # Draw landmarks on the original (unrotated) contour
            # Get PCA center from align_refined_contour_using_pca (mean[0] as center)

            drawDeformitesLandMarks_on_original(contour_image, land_marks, tuple(pca_center), angle)
            
            if structure == required_structure_measurements:
                # Get initial assessments
                deformities_assessments = getAssessments(land_marks, pixel_spacing)
                # Apply Genant grading
                deformities_assessments = applyGenantGrading(deformities_assessments)
                deformities_dict[label] = deformities_assessments
                
    
    return deformities_dict, contour_image

def getDist(point1, point2, pixel_spacing):
    spacing_X, spacing_Y = pixel_spacing  # Extract pixel spacing values
    dist = np.linalg.norm((np.array(point1) - np.array(point2)) * np.array([spacing_X, spacing_Y]))
    # print(dist)
    return dist

def getDistanceFromCornerPoints(corrected_corners,  pixel_spacing):
    
    top_left = corrected_corners[0]
    top_right = corrected_corners[1]
    
    bottm_right = corrected_corners[2]
    bottom_left = corrected_corners[3]
    # Draw raw corners (red)
    dist_left_mm = getDist(bottom_left, top_left, pixel_spacing)
    dist_right_mm= getDist(bottm_right, top_right, pixel_spacing)

    dist_left_mm_str = f"{dist_left_mm:.2f} mm"  # Format to 2 decimal places
    dist_right_mm_str = f"{dist_right_mm:.2f} mm"

    dist_left_mm = np.round(dist_left_mm,3)
    dist_right_mm = np.round(dist_right_mm,3)
    return {"dist_left_mm": dist_left_mm, "dist_right_mm": dist_right_mm} 
    

def draw_corners(image, raw_corners, corrected_corners):
    # Create a copy of the image for display

    # Draw corrected corners (green)
    for point in corrected_corners:
        cv2.circle(image, tuple(point), 2, (0, 255, 255), -1)


    
def drawDistBetweenCorners(image, corrected_corners, pixel_spacing, label):
    # Create a copy of the image for display
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.3
    # font_thickness = 1
    # text_color = (17, 255, 0)  # White

    top_left = corrected_corners[0]
    top_right = corrected_corners[1]
    
    bottm_right = corrected_corners[2]
    bottom_left = corrected_corners[3]
    # Draw raw corners (red)
    # dist_left_mm = getDist(bottom_left, top_left, pixel_spacing)
    # dist_right_mm= getDist(bottm_right, top_right, pixel_spacing)
    # left_vertebrae_height_px = np.linalg.norm((np.array(top_left) - np.array(bottom_left)))
    # right_vertebrae_height_px = np.linalg.norm((np.array(top_right) - np.array(bottm_right)))

    # # Convert distances to strings
    # dist_left_mm_str = f"{dist_left_mm:.2f} mm"  # Format to 2 decimal places
    # dist_right_mm_str = f"{dist_right_mm:.2f} mm"
    
    # distance_text_position_left = (top_left[0]  -50, int(top_left[1] + left_vertebrae_height_px // 2 +5 ))
    # distance_text_position_right = (top_right[0] +30, int(top_right[1] + right_vertebrae_height_px // 2 +5))
    # label_text_position_right =  (top_right[0] +90, int(top_right[1] + right_vertebrae_height_px // 2 +5)) 

    cv2.line(image, tuple(top_left), tuple(bottom_left), (150, 200, 130), 1)  # Cyan for Left
    cv2.line(image, tuple(top_right), tuple(bottm_right), (150, 200, 130), 1)  # Cyan for Left
    





def applyDistanceMeasurementsOnImage(image, mask, pixel_spacing,  required_structure_measurements= "Vertebrae", contour_colors = None, skeletonization= False):
    lumbar_verteberaes = ['S1','L5','L4','L3','L2','L1']
    strucure_labels = {
            'Vertebrae': ['L5','L4','L3','L2','L1','T12','T11','T10'],
            'IVD': ['L5-S1','L4-L5','L3-L4','L2-L3','L1-L2','T12-L1','T11-T12','T10-T11'], 
            'Sacrum': ['S1'] 
        }
    contour_colors = {
        'IVD': (255, 0, 0),
        'Vertebrae': (0, 255, 0),
        'Sacrum': (0, 0, 255)
    }
    measurements_dict = {   } 
    deformities_dict = {   }

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    text_color = (17, 255, 0)  # White

    overlay_image= cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])  # Example structure names
    
    # Draw each structure's contours
    for structure, sorted_contours in contours_dict.items():
        # print(structure)
        if structure == 'IVD':
            last_IVD = LastIVDHolder(True)
        else:
            last_IVD = LastIVDHolder(False)

        if structure =="Vertebrae":
            last_vertebrae = len(sorted_contours)
            # print("vertebras len :" , last_vertebrae)

            
        color = contour_colors.get(structure, (255, 62, 23))  # Default to white
        structure_cls = SegmentationClass[structure].value
        mask_cls = (filtered_mask == structure_cls).astype(np.uint8) * 255
        # plt.imshow(mask_cls)
        for contour, label in zip(sorted_contours[:], strucure_labels[structure]):
            bbox = cv2.boundingRect(contour)
            # Get convex hull points
            corrected_corners, raw_corners = getExtremeCorners(overlay_image, mask_cls, structure, last_IVD, contour, bbox ,skeletonization )
            # Get the position for the text (bottom-right corner of the bounding box)
           
            # cv2.drawContours(overlay_image, [contour], -1, color, thickness=1)
            if structure == required_structure_measurements:
                corner_distance_dict = getDistanceFromCornerPoints(corrected_corners, pixel_spacing)
                measurements_dict[label] = {"corner_points": list(corrected_corners),
                                                                             "distance": corner_distance_dict}
        
                draw_corners(overlay_image, raw_corners, corrected_corners)
                drawDistBetweenCorners(overlay_image, corrected_corners, pixel_spacing,  label) ########## ALERT Problem with Sacrum

    return measurements_dict, overlay_image


def convert_to_builtin_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        # print("np.integer")
        return int(obj)
    elif isinstance(obj, np.floating):
        # print("np.floating")
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # print("np.ndarray")
        return obj.tolist()
    elif isinstance(obj, dict):
        # print("dict")
        return {convert_to_builtin_types(k): convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # print("list or tuple")
        return [convert_to_builtin_types(i) for i in obj]
    else:
        return obj


########################## Angle Measurement ##########################
#######################################################################

def angle_between_vectors(vec1, vec2):
    """Calculate the angle (in degrees) between two vectors using the dot product."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        print("Warning: Zero-length vector detected. Returning default angle (0 degrees).")
        return 0 
        raise ValueError("Zero-length vector provided.")

    # Compute the angle in radians
    theta_rad = np.arccos(dot_product / (norm1 * norm2))
    
    # Convert to degrees
    return np.degrees(theta_rad)

def points_to_vector(p1, p2):
    """Convert two points into a vector."""
    return np.array([p2[0] - p1[0], p2[1] - p1[1]])


def angle_between_lines(p1, p2, p3, p4):
    """Calculate the angle between two lines defined by their endpoints."""
    vec1 = points_to_vector(p1, p2)
    vec2 = points_to_vector(p3, p4)
    return angle_between_vectors(vec1, vec2)

def getSpinalAnglesKeyPoints(label, corrected_corners, key_points_dict):
    if label =="S1":
        top_left, top_right = corrected_corners
        key_points_dict[label]["top_line_points"] = [top_left, top_right]
        
    elif label =="L1":
        top_left, top_right, _, _ = corrected_corners
        key_points_dict[label]["top_line_points"] = [top_left, top_right]
        
    elif label =="L5":
        _, _, bottom_left, bottm_right = corrected_corners
        key_points_dict[label]["bottom_line_points"] =  [bottom_left, bottm_right] 
        
       
def getSpinalAngles(key_points_dict):
    key_points_dict['LLA'] =  getLLA(key_points_dict)
    key_points_dict['LSA'] =  180- getLSA(key_points_dict)
    

def getLLA(key_points_dict):
    L1_top_left = key_points_dict["L1"]["top_line_points"][0]
    L1_top_right = key_points_dict["L1"]["top_line_points"][1]
    S1_top_left = key_points_dict["S1"]["top_line_points"][0]
    S1_top_right = key_points_dict["S1"]["top_line_points"][1]
    angle = angle_between_lines(L1_top_left, L1_top_right, S1_top_left, S1_top_right)

    return angle 
def getLSA(key_points_dict):
    S1_top_left = key_points_dict["S1"]["top_line_points"][0]
    S1_top_right = key_points_dict["S1"]["top_line_points"][1]
    L5_bottom_left = key_points_dict["L5"]["bottom_line_points"][0]
    L5_bottom_right = key_points_dict["L5"]["bottom_line_points"][1]
    angle = angle_between_lines( L5_bottom_left, L5_bottom_right, S1_top_left, S1_top_right)
    return angle 

def drawAngelsKeyPoints(image, key_points_dict, measurements_dict):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    line_color = (255, 0, 0)
    text_color = (17, 255, 0)  # White
    S1_top_left, S1_top_right = key_points_dict["S1"]["top_line_points"]
    cv2.line(image, S1_top_left, S1_top_right, line_color, 2)  

    text_x, text_y = 10, 20  # Starting position for text
    offest_text = 0
    for key, angle in measurements_dict.items():
        if key == 'LLA':
            L1_top_left, L1_top_right = key_points_dict["L1"]["top_line_points"]
            cv2.line(image, L1_top_left, L1_top_right, line_color, 2) 

            
            # cv2.putText(image, f"LLA: {angle:.2f} deg", (text_x, text_y+ offest_text), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        elif key == 'LSA':
            L5_bottom_left, L5_bottom_right = key_points_dict["L5"]["bottom_line_points"]
            cv2.line(image, L5_bottom_left, L5_bottom_right, line_color, 2) 
            # cv2.putText(image, f"LSA: {angle:.2f} deg", (text_x, text_y + offest_text), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            # Draw lines
        offest_text += 20

########################## Spinal Height Measurement ##########################
#######################################################################
# def applySpinalHeightMeasurements(image, mask,  pixel_spacing, 
#                             contour_colors = None, skeletonization= False):
#     spinal_height_key_points= { 

########################## Spinal Height Measurement ##########################
#######################################################################
def applySpinalHeightMeasurements(image, mask,  pixel_spacing, 
                            contour_colors = None, skeletonization= False):
    spinal_height_key_points= { 
            "L1": {
                "superior_point": None,
                "center_point": None,
            },
            "S1": {
                "superior_point": None,
                "center_point": None,
            },
            "superior_hight": None,
            "center_hight": None
    }
    strucure_labels = {
        'Vertebrae': ['L5','L4','L3','L2','L1','T12','T11','T10'],
        'IVD': ['L5-S1','L4-L5','L3-L4','L2-L3','L1-L2','T12-L1','T11-T12','T10-T11'], 
        'Sacrum': ['S1'] 
    }
    
    contour_colors = {
        'IVD': (255, 0, 0),
        'Vertebrae': (0, 255, 0),
        'Sacrum': (0, 0, 255)
    }
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])  # Example structure names
    last_IVD = None
    overlay_image= cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    # Draw each structure's contours
    for structure, sorted_contours in contours_dict.items():
        if structure in ["Vertebrae", "Sacrum"]:
            color = contour_colors.get(structure, (255, 62, 23))  # Default to white
            structure_cls = SegmentationClass[structure].value
            mask_cls = (filtered_mask == structure_cls).astype(np.uint8) * 255
            for contour, label in zip(sorted_contours[:], strucure_labels[structure]):
                if label in ['L1', 'S1' ]:
                    bbox = cv2.boundingRect(contour)
                    # Get convex hull points
                    corrected_corners, raw_corners = getExtremeCorners(overlay_image, mask_cls, structure, last_IVD, contour, bbox ,skeletonization )
                    getSpinalHeightKeyPoint(label, corrected_corners, spinal_height_key_points)

    getSpinalHeight(spinal_height_key_points, pixel_spacing)
    drawSpinalHeight(overlay_image, spinal_height_key_points)
    # print(spinal_height_key_points)

    return spinal_height_key_points, overlay_image
    
def applyAnglesMeasurements(image, mask,  required_angle_measurements= ["LLA", "LSA"],
                            contour_colors = None, skeletonization= False):
    spinal_angel_key_points= { 
    "L1": {
        "top_line_points": [],
    },
    "S1": {
        "top_line_points": [],
    },
    "L5": {
        "bottom_line_points": [],
    },
    "LLA": None,
    "LSA": None
    }
    strucure_labels = {
        'Vertebrae': ['L5','L4','L3','L2','L1','T12','T11','T10'],
        'IVD': ['L5-S1','L4-L5','L3-L4','L2-L3','L1-L2','T12-L1','T11-T12','T10-T11'], 
        'Sacrum': ['S1'] 
    }
    
    measurements_dict = {key: {} for key in required_angle_measurements    } 
    contour_colors = {
        'IVD': (255, 0, 0),
        'Vertebrae': (0, 255, 0),
        'Sacrum': (0, 0, 255)
    }
    last_IVD = None
    overlay_image= cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])  # Example structure names

    # Draw each structure's contours
    for structure, sorted_contours in contours_dict.items():
        if structure in ["Vertebrae", "Sacrum"]:
            color = contour_colors.get(structure, (255, 62, 23))  # Default to white
            structure_cls = SegmentationClass[structure].value
            mask_cls = (filtered_mask == structure_cls).astype(np.uint8) * 255
            # plt.imshow(mask_cls)
            for contour, label in zip(sorted_contours[:], strucure_labels[structure]):
                if label in ['L1', 'L5', 'S1' ]:
                    bbox = cv2.boundingRect(contour)
                    # Get convex hull points
                    corrected_corners, raw_corners = getExtremeCorners(overlay_image, mask_cls, structure, last_IVD, contour, bbox ,skeletonization )
                    getSpinalAnglesKeyPoints(label, corrected_corners, spinal_angel_key_points)

    getSpinalAngles(spinal_angel_key_points)
    for angle in required_angle_measurements:
        measurements_dict[angle] = spinal_angel_key_points[angle]
        
    drawAngelsKeyPoints(overlay_image, spinal_angel_key_points, measurements_dict)
    # print(measurements_dict)
    return measurements_dict, overlay_image 
    
    
def getSpinalHeightKeyPoint(label, corrected_corners, key_points_dict):
    if label =="S1":
        top_left, top_right = corrected_corners
    elif label =="L1":
        top_left, top_right, bottom_left, bottm_right = corrected_corners
        
    key_points_dict[label]["superior_point"] = top_left
    # key_points_dict[label]["center_point"] = (x+ w//2 , y + h//2)

def getSpinalHeight(key_points_dict , pixel_spacing):
    height = getDist(key_points_dict['L1']["superior_point"], key_points_dict['S1']["superior_point"], pixel_spacing)
    key_points_dict["superior_hight"] = f"{height:.2f} mm"
    # key_points_dict["center_hight"] = getDist(key_points_dict['L1']["center_point"], key_points_dict['S1']["center_point"])
    
def drawSpinalHeight(image, key_points_dict):

    heighest_superior_point = key_points_dict['L1']["superior_point"]
    lowest_superior_point = key_points_dict['S1']["superior_point"]

    # text_position = (np.array(key_points_dict['L1']["superior_point"]) + np.array(key_points_dict['S1']["superior_point"]) )//2
    cv2.line(image, tuple(heighest_superior_point), tuple(lowest_superior_point), (150, 200, 130), 1)  # Cyan for Left

########################## Spondylolisthesis Measurement ##########################
###################################################################################

def applySpondylolisithesisClassification(image, mask, 
                            contour_colors = None, skeletonization= False):
    spondylolisithesis_key_points= { 
        "L1": None,
        "L2": None,
        "L3": None,
        "L4": None,
        "L5": None,
        "S1": None     
    }
    lumbar_verteberaes = ['S1','L5','L4','L3','L2','L1']
    strucure_labels = {
        'Vertebrae': ['L5','L4','L3','L2','L1','T12','T11','T10'],
        'IVD': ['L5-S1','L4-L5','L3-L4','L2-L3','L1-L2','T12-L1','T11-T12','T10-T11'], 
        'Sacrum': ['S1'] 
    }
    contour_colors = {
        'IVD': (255, 0, 0),
        'Vertebrae': (0, 255, 0),
        'Sacrum': (0, 0, 255)
    }
    last_IVD = None
    overlay_image= cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])  # Example structure names

    # Draw each structure's contours
    for structure, sorted_contours in contours_dict.items():
        if structure in ["Vertebrae", "Sacrum"]:
            color = contour_colors.get(structure, (255, 62, 23))  # Default to white
            structure_cls = SegmentationClass[structure].value
            mask_cls = (filtered_mask == structure_cls).astype(np.uint8) * 255
            for contour, label in zip(sorted_contours[:], strucure_labels[structure]):
                if label in lumbar_verteberaes:
                    bbox = cv2.boundingRect(contour)
                    # Get convex hull points
                    corrected_corners, raw_corners = getExtremeCorners(overlay_image, mask_cls, structure, last_IVD, contour, bbox ,skeletonization )
                    getcentroid(label, corrected_corners, spondylolisithesis_key_points)

    classification_dict = classify_spondylolisthesis(spondylolisithesis_key_points)
    draw_centroid_lines(overlay_image, spondylolisithesis_key_points)

    return classification_dict, overlay_image

def classify_spondylolisthesis(spinal_disorder_disease_key_points):
    """Classifies vertebral dislocation based on angular deviation."""
    Vertrical_line_length = 30
    centroids = [spinal_disorder_disease_key_points[vb] for vb in spinal_disorder_disease_key_points if spinal_disorder_disease_key_points[vb] is not None]
    
    angles = [calculate_angle(centroids[i], centroids[i+1], Vertrical_line_length) for i in range(len(centroids)-1)]
    theta_mean = np.mean(angles)
    theta_std = np.std(angles)
    
    classifications = {}
    for i, theta in enumerate(angles):
        if theta < (theta_mean - theta_std):
            classification = "Posterior Displacement"
        elif theta > (theta_mean + theta_std):
            classification = "Anterior Displacement"
        else:
            classification = "Normal"
        classifications[f"L{i+1}"] = {"centroid":centroids[i],"Angle": f"{theta:.2f}", "Classification": classification}
    
    return classifications

def draw_centroid_lines(image, spinal_disorder_disease_key_points):
    """Draws lines between centroids and vertical reference lines."""
    # print("draw_centroid_lines")
    # print("centroid", spinal_disorder_disease_key_points)
    
    for i, vb in enumerate(spinal_disorder_disease_key_points):
        if spinal_disorder_disease_key_points[vb] is None:
            continue
        centroid = spinal_disorder_disease_key_points[vb]
        # print("centroid", centroid)
        cv2.circle(image, centroid, 2, (87, 144, 200), -1)  # Red dot at centroid      
        
        if i < len(spinal_disorder_disease_key_points) - 1:
            next_vb = list(spinal_disorder_disease_key_points.keys())[i + 1]
            if spinal_disorder_disease_key_points[next_vb] is not None:
                next_centroid = spinal_disorder_disease_key_points[next_vb]
                cv2.line(image, centroid, next_centroid, (0, 255, 255), 2)  # Yellow line between centroids
            cv2.line(image, (centroid[0], centroid[1]), (centroid[0], centroid[1] + 30), (0, 255, 0), 2)  # Vertical green line
    
    # return image

def draw_classification_results(image, classifications):
    """Draws classification results as text on the top left side of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    line_color = (255, 0, 0)
    text_color = (17, 255, 0)  # White
    y_offset = 20
    for vb, result in classifications.items():
        text = f"{vb}: {result['Angle']} deg -> {result['Classification']}"
        cv2.putText(image, text, (10, y_offset), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        y_offset += 20

def getcentroid(label, corrected_corners, key_points_dict):
    vertebrae_centroid = None
    if label =="S1":
        top_left, top_right = corrected_corners
        vertebrae_centroid = verterbraeMidPoint(top_left, top_right)
    else:
        top_left, top_right, bottom_left, bottm_right = corrected_corners
        vertebrae_centroid = vertebraeCentroid(top_left, top_right, bottom_left, bottm_right)
    key_points_dict[label] = vertebrae_centroid

def verterbraeMidPoint(p1, p2):
    """Computes the midpoint between two points in 2D space."""
    # return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    return (round(p2[0]*0.980), round(p1[1]*0.980) )
    

def vertebraeCentroid(p1, p2, p3, p4):
    """Computes the centroid of four points in 2D space."""
    cx = (p1[0] + p2[0] + p3[0] + p4[0]) // 4
    cy = (p1[1] + p2[1] + p3[1] + p4[1]) // 4
    return (cx, cy)

def calculate_angle(vb1, vb2, Vertrical_line_length):
    """Calculates the angle between the vertical line from vb1 and the line vb1 -> vb2."""
    vec_v = np.array([0, Vertrical_line_length])  # Vertical reference vector
    vec_vb = np.array([vb2[0] - vb1[0], vb2[1] - vb1[1]])
    
    dot_product = np.dot(vec_v, vec_vb)
    norm_v = np.linalg.norm(vec_v)
    norm_vb = np.linalg.norm(vec_vb)
    
    cos_theta = dot_product / (norm_v * norm_vb)
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Clip to avoid numerical errors
    
    return theta


##########################Lumbar Lordosis Angle##########################
#######################################################################
def applyLumbarLordosisClassification(image, mask, pixel_spacing, 
                            contour_colors = None, skeletonization= False):
    lumbar_lordosis_key_points= { 
        "L1": None,
        "L2": None,
        "L3": None,
        "L4": None,
        "L5": None,
        "S1": None     
    }
    lumbar_verteberaes = ['S1','L5','L4','L3','L2','L1']
    strucure_labels = {
        'Vertebrae': ['L5','L4','L3','L2','L1','T12','T11','T10'],
        'IVD': ['L5-S1','L4-L5','L3-L4','L2-L3','L1-L2','T12-L1','T11-T12','T10-T11'], 
        'Sacrum': ['S1'] 
    }
    contour_colors = {
        'IVD': (255, 0, 0),
        'Vertebrae': (0, 255, 0),
        'Sacrum': (0, 0, 255)
    }
    
    last_IVD = None
    overlay_image= cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        
    mask_classes = convertToCategoricalMasks(mask)
    filtered_mask = getIVDAndPEClass(mask_classes)
    contours_dict = getContours(filtered_mask, structure_names=["IVD", "Vertebrae", "Sacrum"])
    # Draw each structure's contours
    for structure, sorted_contours in contours_dict.items():
        if structure in ["Vertebrae", "Sacrum"]:
            color = contour_colors.get(structure, (255, 62, 23))  # Default to white
            structure_cls = SegmentationClass[structure].value
            mask_cls = (filtered_mask == structure_cls).astype(np.uint8) * 255
            for contour, label in zip(sorted_contours[:], strucure_labels[structure]):
                if label in lumbar_verteberaes:
                    bbox = cv2.boundingRect(contour)
                    # Get convex hull points
                    corrected_corners, raw_corners = getExtremeCorners(overlay_image, mask_cls, structure, last_IVD, contour, bbox ,skeletonization )
                    getcentroid(label, corrected_corners, lumbar_lordosis_key_points)
                    
    lordosis_classification = classify_lordosis(lumbar_lordosis_key_points, overlay_image, pixel_spacing)
    draw_enclosed_area(overlay_image, lumbar_lordosis_key_points )
    return lordosis_classification, overlay_image

def classify_lordosis(spinal_disorder_disease_key_points, image, pixel_spacing, phi=(39, 53)):
    """Classifies lordosis based on area range (phi) adjusted to match clinical criteria."""
    centroids= compute_centroids(spinal_disorder_disease_key_points)
    # centroids_curve = centroidsInterpolation(centroids)
    # image = draw_enclosed_area(image, centroids_curve )
    AR = compute_enclosed_area(image, centroids, pixel_spacing)
    if AR > phi[1]:
        return f"Hyper-lordosis: {AR}"
    elif AR < phi[0]:
        return f"Hypo-lordosis: {AR}"
    else:
        return f"Normal lordosis: {AR}"


def compute_centroids(spinal_disorder_disease_key_points):
    """Extracts centroids from the provided dictionary."""
    centroids  = [spinal_disorder_disease_key_points[vb] for vb in spinal_disorder_disease_key_points if spinal_disorder_disease_key_points[vb] is not None]
        # Convert to numpy array and sort by x-coordinates
    # print(centroids)
    # centroids = np.array(sorted(centroids, key=lambda pt: pt[0]))
    centroids = centroidsInterpolation(centroids)
    return centroids

def centroidsInterpolation(centroids):
    """Fits a cubic spline to the spinal curve, handling vertical alignment."""
    centroids = np.array(centroids)

    if len(centroids) < 2:
        return centroids.astype(np.int32)

    # Extract x and y values
    x, y = centroids[:, 0], centroids[:, 1]

    # CubicSpline requires the x-axis input to be strictly increasing. In this
    # function y is used as that axis, so sort top-to-bottom and merge duplicate
    # rows that can appear after integer centroid rounding.
    order = np.argsort(y)
    x = x[order]
    y = y[order]

    unique_y = []
    unique_x = []
    for y_value in np.unique(y):
        same_row = y == y_value
        unique_y.append(y_value)
        unique_x.append(np.mean(x[same_row]))

    y = np.array(unique_y, dtype=np.float64)
    x = np.array(unique_x, dtype=np.float64)

    if len(y) < 2:
        return np.column_stack([x, y]).astype(np.int32)

    # Apply cubic spline interpolation
    cs = CubicSpline(y, x)
    y_new = np.linspace(y.min(), y.max(), 100)
    x_new = cs(y_new)

    # Swap back if needed
    # if swapped:
        # x_new, y_new = y_new, x_new  # Swap back to original order

    return np.array(list(zip(x_new, y_new)), dtype=np.int32)

def compute_enclosed_area(image, centroids, pixel_spacing):
    """Computes the enclosed area (AR) between the spinal curve and the vertical projection."""

    AR = 0
    # Loop through each segment formed by centroids
    for i in range(1, len(centroids)):
        p_i = np.abs(centroids[i][0] - centroids[i-1][0])  # Horizontal distance between centroids (p_i)
        delta_p = centroids[i][1] - centroids[i-1][1]  # Vertical difference between centroids (Δp)
        
        # Update AR with area of trapezoid (p_i * Δp)
        AR += p_i *pixel_spacing[0] *  delta_p * pixel_spacing[1]  # Multiply by pixel_spacing if necessary to get real area

    return round(AR)



def draw_enclosed_area(image, spinal_disorder_disease_key_points):
    """Draws the enclosed area (AR) between the spinal curve and the vertical projection."""
    centroids = compute_centroids(spinal_disorder_disease_key_points)
    # Close the curve by projecting the first and last centroids vertically
    # Get the min and max Y-values for projection
    y_min, y_max = centroids[:, 1].min(), centroids[:, 1].max()
    bottom_projection = [(centroids[0][0], y_min)]
    
    full_contour = np.vstack([centroids, bottom_projection])  # Stack to form a closed shape

    # Create mask and fill the polygon
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [full_contour.reshape(-1, 1, 2)], (0,0,255))

    area_pixels = np.any(mask > 0, axis=2)
    blended = cv2.addWeighted(image, 0.75, mask, 0.25, 0)
    image[area_pixels] = blended[area_pixels]


def applySagittalMeasurements(sagittal_mid_slice, Mid_slice_mask, patient_id):
    """Apply all sagittal measurements and save visualizations
    """


    VIEWE_PLANE = "sagittal"
    PIX_SPACING_X, PIX_SPACING_Y = sagittal_mid_slice.PixelSpacing
    image_uint16 = sagittal_mid_slice.pixel_array
    image = uint16Touint8(image_uint16)
    # Get prediction mask
    pred_mask =  Mid_slice_mask
    
    # Display segmentation overlay
    image_segemtnation_overlay = getImageSegmentationOverlay(image_uint16, pred_mask, VIEWE_PLANE)
    segmentation_overlay_path = display_image(
        image_segemtnation_overlay,
        "Segmentation Overlay",
        patient_id=patient_id,
        function_name="sagittal_segmentation"
    )
    Vertebrae_deformities_measurements_dict, Vertebrae_deformities_overlay_image = applyVertebraeDeformitiesAssessmentsOnImage(image, pred_mask, 
                                                                      pixel_spacing=[PIX_SPACING_X, PIX_SPACING_Y],  patient_id=patient_id,
                                                                              required_structure_measurements = "Vertebrae",  skeletonization=False)
    
    
    # print("Vertebrae Deformities Measurements:")
    # print(Vertebrae_deformities_measurements_dict)
    vertebrae_deformities_path = display_image(
        Vertebrae_deformities_overlay_image,
        "Vertebrae Deformities Measurements",
        patient_id=patient_id,
        function_name="vertebrae_deformities"
    )

    # Vertebrae height measurements
    Vertebrae_height_measurements_dict, Vertebrae_height_overlay_image = applyDistanceMeasurementsOnImage(
        image, pred_mask, 
        pixel_spacing=[PIX_SPACING_X, PIX_SPACING_Y],
        required_structure_measurements="Vertebrae",
        skeletonization=False
    )

    # Convert measurements to Python native types
    Vertebrae_height_measurements_dict = convert_to_builtin_types(Vertebrae_height_measurements_dict)
    vertebrae_height_path = display_image(
        Vertebrae_height_overlay_image,
        "Vertebrae Height Measurements",
        patient_id=patient_id,
        function_name="vertebrae_height"
    )
    
    # IVD height measurements
    IVD_height_measurements_dict, IVD_height_overlay_image = applyDistanceMeasurementsOnImage(
        image, pred_mask,
        pixel_spacing=[PIX_SPACING_X, PIX_SPACING_Y],
        required_structure_measurements="IVD",
        skeletonization=False
    )
    # Convert measurements to Python native types
    IVD_height_measurements_dict = convert_to_builtin_types(IVD_height_measurements_dict)
    ivd_height_path = display_image(
        IVD_height_overlay_image,
        "IVD Height Measurements",
        patient_id=patient_id,
        function_name="ivd_height"
    )
    
    # Angle measurements
    angle_measurements_dict, angle_overlay_image = applyAnglesMeasurements(image, pred_mask)
    # Convert measurements to Python native types
    angle_measurements_dict = convert_to_builtin_types(angle_measurements_dict)
    angle_measurements_path = display_image(
        angle_overlay_image,
        "Angle Measurements",
        patient_id=patient_id,
        function_name="angle_measurements"
    )
    
    # Spinal height measurements
    height_measurements_dict, height_overlay_image = applySpinalHeightMeasurements(
        image, pred_mask,
        pixel_spacing=[PIX_SPACING_X, PIX_SPACING_Y]
    )
    # Convert measurements to Python native types
    height_measurements_dict = convert_to_builtin_types(height_measurements_dict)
    spinal_height_path = display_image(
        height_overlay_image,
        "Spinal Height Measurements",
        patient_id=patient_id,
        function_name="spinal_height"
    )
    
    # Spondylolisthesis classification
    spondlylisithesis_classification, spondlylisithesis_overlay_image = applySpondylolisithesisClassification(
        image,
        pred_mask
    )
    # Convert measurements to Python native types
    spondlylisithesis_classification = convert_to_builtin_types(spondlylisithesis_classification)

    spondylolisthesis_path = display_image(
        spondlylisithesis_overlay_image,
        "Spondylolisthesis Classification",
        patient_id=patient_id,
        function_name="spondylolisthesis"
    )
    # Lumbar Lordosis Classification
    lordosis_classification, lordosis_overlay_image = applyLumbarLordosisClassification(
        image,
        pred_mask,
        pixel_spacing=[PIX_SPACING_X, PIX_SPACING_Y]

    )
    # Convert measurements to Python native types
    lordosis_classification = convert_to_builtin_types(lordosis_classification)

    lumbar_lordosis_path = display_image(
        lordosis_overlay_image,
        "Lumbar Lordosis Classification",
        patient_id=patient_id,
        function_name="lumbar_lordosis"
    )
    
    # Encode images for API response
    _, pred_mask_encoded = cv2.imencode(".png", pred_mask)
    pred_mask_encoded_base64 = base64.b64encode(pred_mask_encoded.tobytes()).decode("utf-8")
    
    return {
        "pred_mask": pred_mask_encoded_base64,
        "segmentation_overlay": {
            "measurements": None,
            "image_overlay": segmentation_overlay_path
        },
        "vertebrae_height": {
            "measurements": Vertebrae_height_measurements_dict,
            "image_overlay": vertebrae_height_path
        },
        "ivd_height": {
            "measurements": IVD_height_measurements_dict,
            "image_overlay": ivd_height_path
        },
        "vertebrae_deformities": {
            "measurements_info": convert_to_builtin_types(Vertebrae_deformities_measurements_dict),
            "image_overlay": vertebrae_deformities_path,
        },
        "angles": {
            "measurements": angle_measurements_dict,
            "image_overlay": angle_measurements_path
        },
        "spinal_height": {
            "measurements": height_measurements_dict,
            "image_overlay": spinal_height_path
        },
        "spondylolisthesis": {
            "measurements": spondlylisithesis_classification,
            "image_overlay": spondylolisthesis_path
        },
        "lumbar_lordosis": {
            "measurements": lordosis_classification,
            "image_overlay": lumbar_lordosis_path
        }
    }

def getAllMeasurements(sagittal_mid_slice, Mid_slice_mask, patient_id):
    " input the segementation mask of the middle sagittal slice and return the measurements and diagnosis "


    sagittal_diagnosis = applySagittalMeasurements(sagittal_mid_slice, Mid_slice_mask, patient_id)



    return sagittal_diagnosis
    
    



