import os
import cv2
import natsort
import numpy as np

def load_images_from_folder(folder_path, extensions=None):
    """
    Loads images sequentially from a specified folder.

    Args:
        folder_path (str): The path to the folder containing images.
        extensions (list, optional): A list of allowed image file extensions
                                     (e.g., ['.jpg', '.png']).
                                     Defaults to ['.jpg', '.jpeg', '.png', '.tiff'].

    Yields:
        tuple: A tuple containing:
            - filename (str): The name of the loaded image file.
            - image (numpy.ndarray): The loaded image as a NumPy array (BGR format).
                                     Returns None for the image if loading fails.

    Raises:
        FileNotFoundError: If the specified folder_path does not exist.
        ValueError: If no images are found in the specified folder.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Error: The folder '{folder_path}' was not found.")

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']

    image_files = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_files.append(filename)

    if not image_files:
        raise ValueError(f"No images with supported extensions found in '{folder_path}'.")

    # Sort files alphanumerically.
    image_files = natsort.natsorted(image_files) # Natural sort

    print(f"Found {len(image_files)} images. Processing order:")
    for i, filename in enumerate(image_files):
        print(f"  {i+1}. {filename}")

    for filename in image_files:
        file_path = os.path.join(folder_path, filename)
        try:
            image = cv2.imread(file_path)
            if image is None:
                print(f"Warning: Could not load image {filename}. Skipping.")
                yield filename, None
            else:
                yield filename, image
        except Exception as e:
            print(f"Error loading image {filename}: {e}. Skipping.")
            yield filename, None


def extract_features(image, image_filename=""):
    """
    Detects SIFT keypoints and computes their descriptors.

    Args:
        image (numpy.ndarray): The input image (BGR format).
        image_filename (str, optional): Name of the image file for logging.

    Returns:
        tuple: A tuple containing:
            - keypoints (list of cv2.KeyPoint): Detected keypoints.
            - descriptors (numpy.ndarray): Computed SIFT descriptors (N x 128).
                                           Returns (None, None) if image is invalid.
    """
    if image is None:
        print(f"Feature extraction: Invalid image provided ({image_filename}). Skipping.")
        return None, None

    # Convert image to grayscale as SIFT works on intensity values
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a SIFT object
    # tune params like nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma?
    sift = cv2.SIFT_create() # Using default parameters

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    if descriptors is None: # Handles cases where no features are found
        descriptors = np.array([]) # Return empty array instead of None for consistency
        
    print(f"Feature extraction: Found {len(keypoints)} keypoints in {image_filename if image_filename else 'image'}.")

    return keypoints, descriptors

def match_features(descriptors1, descriptors2, ratio_thresh=0.75):
    """
    Matches features between two sets of descriptors using FLANN, Lowe's ratio test,
    and cross-checking.

    Args:
        descriptors1 (numpy.ndarray): Descriptors for the first image.
        descriptors2 (numpy.ndarray): Descriptors for the second image.
        ratio_thresh (float): Lowe's ratio test threshold.

    Returns:
        list of cv2.DMatch: A list of good, cross-checked matches.
                           The DMatch objects will have queryIdx referring to descriptors1
                           and trainIdx referring to descriptors2.
    """
    if descriptors1 is None or descriptors2 is None or descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
        print("  Matching: Not enough descriptors to match.")
        return []

    # Ensure descriptors are float32, as FLANN expects this for SIFT descriptors
    if descriptors1.dtype != np.float32:
        descriptors1 = np.float32(descriptors1)
    if descriptors2.dtype != np.float32:
        descriptors2 = np.float32(descriptors2)

    # FLANN parameters for SIFT/SURF (float descriptors)
    FLANN_INDEX_KDTREE = 1  # Algorithm type: k-d tree
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # Higher checks = more accuracy, more time

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Helper function to apply knnMatch and Lowe's ratio test
    def get_good_matches_via_knn(desc1, desc2):
        knn_matches = flann.knnMatch(desc1, desc2, k=2)
        
        current_good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2: # Ensure two neighbors were found
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    current_good_matches.append(m)
        return current_good_matches

    # Match descriptors1 to descriptors2
    # good_matches1to2 are DMatch objects where queryIdx is from desc1, trainIdx is from desc2
    good_matches1to2 = get_good_matches_via_knn(descriptors1, descriptors2)


    # Match descriptors2 to descriptors1
    # good_matches2to1 are DMatch objects where queryIdx is from desc2, trainIdx is from desc1
    good_matches2to1 = get_good_matches_via_knn(descriptors2, descriptors1)
 

    # Cross-checking
    cross_checked_matches = []
    # Create a lookup for faster cross-checking.
    # For matches2to1, queryIdx is an index in descriptors2, trainIdx is an index in descriptors1.
    # We want to find if for a match (idx1_d1, idx2_d2) from good_matches1to2,
    # there is a match (idx2_d2, idx1_d1) in good_matches2to1.
    
    lookup_matches2to1 = {} # Key: (desc2_idx, desc1_idx), Value: DMatch object
    for m in good_matches2to1:
        lookup_matches2to1[(m.queryIdx, m.trainIdx)] = m

    for m1to2 in good_matches1to2:
        # m1to2.queryIdx is index in descriptors1
        # m1to2.trainIdx is index in descriptors2
        # We are looking for a match in good_matches2to1 where:
        #   queryIdx (from desc2) == m1to2.trainIdx
        #   trainIdx (from desc1) == m1to2.queryIdx
        if (m1to2.trainIdx, m1to2.queryIdx) in lookup_matches2to1:
            cross_checked_matches.append(m1to2) # Keep the DMatch object from the 1->2 perspective

    return cross_checked_matches

def parse_colmap_camera_file(filepath):
    """
    Parses a COLMAP cameras.txt file.
    For simplicity, assumes a single camera model if multiple are listed,
    and specifically looks for SIMPLE_RADIAL or PINHOLE models.

    Args:
        filepath (str): Path to the cameras.txt file.

    Returns:
        dict: A dictionary containing camera parameters like
              f (focal_length), cx, cy, k1, and the camera_matrix K.
              Returns None if parsing fails or model is not supported.
    """
    cameras = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            
            # For SIMPLE_RADIAL: f, cx, cy, k
            # For PINHOLE: fx, fy, cx, cy
            
            params_dict = {"id": camera_id, "model": model, "width": width, "height": height}
            
            if model == "SIMPLE_RADIAL" and len(parts) >= 8:
                focal_length = float(parts[4])
                cx = float(parts[5])
                cy = float(parts[6])
                k1 = float(parts[7])
                # k2 = float(parts[8]) if len(parts) > 8 else 0.0 # If SIMPLE_RADIAL_FISHEYE or similar
                
                params_dict.update({"f": focal_length, "cx": cx, "cy": cy, "k1": k1})
                K = np.array([[focal_length, 0, cx],
                              [0, focal_length, cy],
                              [0, 0, 1]])
                params_dict["K"] = K
                cameras[camera_id] = params_dict
                return params_dict # Return the first successfully parsed camera relevant to us

            elif model == "PINHOLE" and len(parts) >= 8:
                fx = float(parts[4])
                fy = float(parts[5])
                cx = float(parts[6])
                cy = float(parts[7])

                params_dict.update({"fx": fx, "fy": fy, "cx": cx, "cy": cy})
                K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
                params_dict["K"] = K
                cameras[camera_id] = params_dict
                return params_dict # Return the first successfully parsed camera

    if not cameras:
        print(f"Warning: No supported camera model (SIMPLE_RADIAL or PINHOLE) found in {filepath}")
        return None
    
    # If multiple cameras are found, return the first one
    return list(cameras.values())[0] if cameras else None
