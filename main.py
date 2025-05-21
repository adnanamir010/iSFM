import numpy as np
import cv2
from utils import load_images_from_folder, extract_features, match_features, parse_colmap_camera_file
import os

# --- Configuration ---
image_folder_path = 'data/images'
camera_file_path = 'data/cameras.txt'
sequential_match_window = 1
ransac_threshold_pixels = 1.0 # RANSAC threshold for findEssentialMat (in pixels)
ransac_confidence = 0.999    # RANSAC confidence for findEssentialMat
min_inlier_count_for_pair = 15 # Minimum number of inliers to consider a pair valid

# --- Load Camera Intrinsics ---
print(f"\n--- Loading Camera Intrinsics from: {camera_file_path} ---")
camera_intrinsics = parse_colmap_camera_file(camera_file_path)
K_matrix = None
if camera_intrinsics and 'K' in camera_intrinsics:
    K_matrix = camera_intrinsics['K']
    print(f"Loaded camera model: {camera_intrinsics.get('model')}")
    print(f"Intrinsic Matrix K:\n{K_matrix}")
else:
    print("Critical Error: Could not load camera intrinsics matrix K. Cannot proceed with geometric verification.")
    exit()

# --- 1. Load Images and Extract Features ---
print(f"\n--- Loading Images & Extracting Features from: {os.path.abspath(image_folder_path)} ---")
# This will store features and other per-image data
# { "filename": {"keypoints": ..., "descriptors": ..., "image_data": ...}, ...}
all_image_data = {} 
# Keep an ordered list of filenames for sequential processing
ordered_filenames = []

try:
    image_loader = load_images_from_folder(image_folder_path)
    first_image_visualized = False # Simpler flag for one-time visualization

    for filename, image_data_cv2 in image_loader:
        if image_data_cv2 is not None:
            ordered_filenames.append(filename)
            print(f"Processing {filename} for feature extraction...")
            keypoints, descriptors = extract_features(image_data_cv2, filename) # utils.extract_features

            current_image_info = {
                "keypoints": keypoints,
                "descriptors": descriptors,
                "image_data": image_data_cv2 # Storing image data for visualization
            }
            all_image_data[filename] = current_image_info
            
            if keypoints is not None and len(keypoints) > 0 and descriptors is not None:
                print(f"  Stored {len(keypoints)} keypoints and descriptors of shape {descriptors.shape}")

                # Optional: Visualize keypoints on the first image with features
                if not first_image_visualized:
                    img_with_keypoints = cv2.drawKeypoints(image_data_cv2, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.imshow(f"Keypoints on {filename}", img_with_keypoints)
                    cv2.waitKey(1000) # Display for 1 second
                    cv2.destroyAllWindows()
                    first_image_visualized = True
            else:
                print(f"  No features extracted or error for {filename}.")
        else:
            print(f"Skipping feature extraction for {filename} as it could not be loaded.")

    print("\n--- Feature Extraction Summary ---")
    for filename, data in all_image_data.items():
        if data["descriptors"] is not None:
            print(f"Image: {filename}, Keypoints: {len(data['keypoints'])}, Descriptors shape: {data['descriptors'].shape}")
        else:
            print(f"Image: {filename}, Keypoints: {len(data['keypoints'])}, Descriptors: None")


    # --- 2. Perform Sequential Feature Matching ---
    print("\n--- Performing Sequential Feature Matching ---")
    # Stores matches: {(img_name1, img_name2): [cv2.DMatch_objects]}
    all_matches_info = {}
    
    visualized_one_match_pair = False

    for i in range(len(ordered_filenames)):
        for k in range(1, sequential_match_window + 1):
            if i + k < len(ordered_filenames):
                filename1 = ordered_filenames[i]
                filename2 = ordered_filenames[i + k]

                print(f"Attempting to match features between '{filename1}' and '{filename2}'...")

                data1 = all_image_data.get(filename1)
                data2 = all_image_data.get(filename2)

                if data1 and data2 and \
                   data1["descriptors"] is not None and data2["descriptors"] is not None and \
                   len(data1["keypoints"]) > 0 and len(data2["keypoints"]) > 0:
                    
                    good_matches = match_features(data1["descriptors"], data2["descriptors"], ratio_thresh=0.75)
                    
                    pair_key = tuple(sorted((filename1, filename2))) # Use sorted tuple to make key canonical
                    
                    if good_matches:
                         all_matches_info[pair_key] = good_matches
                         print(f"  Found {len(good_matches)} good matches between {filename1} and {filename2}.")

                         # Optional: Visualize matches for the first pair that has matches
                         if not visualized_one_match_pair and camera_intrinsics: # Only visualize if we have K for context
                             img1_to_draw = data1["image_data"]
                             kp1 = data1["keypoints"]
                             img2_to_draw = data2["image_data"]
                             kp2 = data2["keypoints"]
                             
                             img_matches_vis = cv2.drawMatches(img1_to_draw, kp1, img2_to_draw, kp2, good_matches, None, 
                                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                             
                             # Resize for display if too large
                             max_display_width = 1280
                             if img_matches_vis.shape[1] > max_display_width:
                                 scale_factor = max_display_width / img_matches_vis.shape[1]
                                 width = int(img_matches_vis.shape[1] * scale_factor)
                                 height = int(img_matches_vis.shape[0] * scale_factor)
                                 resized_img_matches = cv2.resize(img_matches_vis, (width, height), interpolation=cv2.INTER_AREA)
                             else:
                                 resized_img_matches = img_matches_vis
                             
                             cv2.imshow(f"Matches between {filename1} and {filename2}", resized_img_matches)
                             cv2.waitKey(1000) # Display for 1 second
                             cv2.destroyAllWindows()
                             visualized_one_match_pair = True
                    else:
                        print(f"  No good matches found between {filename1} and {filename2}.")

                else:
                    desc1_info = f"({len(data1['descriptors']) if data1 and data1['descriptors'] is not None else 'N/A'} descriptors)" if data1 else "(Data N/A)"
                    desc2_info = f"({len(data2['descriptors']) if data2 and data2['descriptors'] is not None else 'N/A'} descriptors)" if data2 else "(Data N/A)"
                    print(f"  Not enough descriptors/keypoints to match for {filename1} {desc1_info} or {filename2} {desc2_info}.")


    print("\n--- Feature Matching Summary ---")
    if all_matches_info:
        for (img_name1, img_name2), matches_list in all_matches_info.items():
            print(f"Pair: ({img_name1}, {img_name2}), Good Matches: {len(matches_list)}")
    else:
        print("No matches found between any pairs.")


except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()

if 'all_matches_info' not in locals(): # Check if populated
    print("Warning: 'all_matches_info' not populated. Geometric verification requires matches.")
    all_matches_info = {}


# --- 3. Perform Geometric Verification ---
print("\n--- Performing Geometric Verification ---")
# Stores verified pairs: {(filename1, filename2): {"matches": ..., "E": ..., "R": ..., "t": ..., "pts1": ..., "pts2": ...}}
verified_view_pairs = {}
visualization_done_geom_verify = False

for pair_key, good_DMatches in all_matches_info.items():
    filename1, filename2 = pair_key

    # Ensure we have enough matches for the 5-point algorithm in findEssentialMat
    if len(good_DMatches) < 8: # Technically 5, but more is better for RANSAC stability
        print(f"  Pair {pair_key}: Skipped - Not enough initial matches ({len(good_DMatches)}). Need at least 8.")
        continue

    # Retrieve keypoints for the current pair
    kp1 = all_image_data[filename1]["keypoints"]
    kp2 = all_image_data[filename2]["keypoints"]
    img1_data = all_image_data[filename1]["image_data"] # For visualization
    img2_data = all_image_data[filename2]["image_data"] # For visualization


    # Extract 2D point coordinates from the DMatch objects
    # These points should be in the coordinate system consistent with K_matrix
    # (i.e., if features were extracted on original images, K_matrix is original K)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_DMatches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_DMatches]).reshape(-1, 1, 2)

    # Estimate Essential Matrix using RANSAC
    # Note: In OpenCV 4.x, 'cameraMatrix' parameter is named 'K' for findEssentialMat
    E, mask_E = cv2.findEssentialMat(pts1, pts2, K=K_matrix, method=cv2.RANSAC, 
                                     prob=ransac_confidence, threshold=ransac_threshold_pixels)

    if E is None or mask_E is None:
        print(f"  Pair {pair_key}: Skipped - Essential Matrix could not be computed.")
        continue
    
    num_inliers_E = np.sum(mask_E)
    # print(f"  Pair {pair_key}: Essential Matrix RANSAC resulted in {num_inliers_E} inliers (out of {len(good_DMatches)}).")

    if num_inliers_E < min_inlier_count_for_pair:
        print(f"  Pair {pair_key}: Skipped - Too few inliers ({num_inliers_E}) after E-Matrix RANSAC. Minimum required: {min_inlier_count_for_pair}.")
        continue

    # Filter points and DMatches using the RANSAC mask from findEssentialMat
    pts1_inliers_for_E = pts1[mask_E.ravel() == 1]
    pts2_inliers_for_E = pts2[mask_E.ravel() == 1]
    dmatches_inliers_for_E = [d for i, d in enumerate(good_DMatches) if mask_E[i,0] == 1]

    # Recover Pose (R, t) from Essential Matrix
    # The points passed to recoverPose must be the inliers from findEssentialMat
    # Note: In OpenCV 4.x, 'cameraMatrix' parameter is named 'K' for recoverPose
    retval_num_points, R, t, mask_RP = cv2.recoverPose(E, pts1_inliers_for_E, pts2_inliers_for_E, K=K_matrix)
                                                       # Optional 'mask' output for recoverPose in some versions,
                                                       # here mask_RP is the mask of inliers satisfying chirality.

    if R is None or t is None: # Should not happen if E was valid. retval_num_points would be 0.
        print(f"  Pair {pair_key}: Skipped - Pose recovery (R,t) failed from E matrix.")
        continue

    num_inliers_RP = retval_num_points # This is the count of points satisfying chirality
    # print(f"  Pair {pair_key}: Pose recovered. Chirality check resulted in {num_inliers_RP} valid points.")

    if num_inliers_RP < min_inlier_count_for_pair:
        print(f"  Pair {pair_key}: Skipped - Too few inliers ({num_inliers_RP}) after pose recovery and chirality check. Minimum required: {min_inlier_count_for_pair}.")
        continue
        
    # Filter DMatches and points again using the mask from recoverPose (mask_RP)
    # mask_RP applies to pts1_inliers_for_E and pts2_inliers_for_E
    final_verified_dmatches = []
    final_pts1_coords = [] # For storing (N,2) coordinate arrays
    final_pts2_coords = []

    if mask_RP is not None: # mask_RP indicates points satisfying chirality among those passed to recoverPose
        for i in range(len(dmatches_inliers_for_E)):
            if mask_RP[i, 0] == 1: # If this DMatch's points satisfy chirality
                final_verified_dmatches.append(dmatches_inliers_for_E[i])
                # Store the corresponding 2D points (already reshaped by .ravel() earlier if needed)
                final_pts1_coords.append(pts1_inliers_for_E[i].ravel()) 
                final_pts2_coords.append(pts2_inliers_for_E[i].ravel())
    
    if not final_verified_dmatches:
        print(f"  Pair {pair_key}: Skipped - No inliers left after chirality check applied to DMatches.")
        continue
    
    num_final_inliers = len(final_verified_dmatches)
    print(f"  Pair {pair_key}: VERIFIED with E and R,t. Final inliers: {num_final_inliers}.")

    # Store the verified data
    verified_view_pairs[pair_key] = {
        "matches": final_verified_dmatches,       # List of cv2.DMatch objects
        "pts1": np.array(final_pts1_coords),      # (N,2) numpy array of points from image1
        "pts2": np.array(final_pts2_coords),      # (N,2) numpy array of points from image2
        "E": E,                                   # Essential Matrix
        "R": R,                                   # Rotation Matrix
        "t": t,                                   # Translation Vector (unit scale)
        "num_inliers": num_final_inliers
    }

    # Optional: Visualize inlier matches for the first successfully verified pair
    if not visualization_done_geom_verify and final_verified_dmatches:
        img_matches_verified = cv2.drawMatches(img1_data, kp1, img2_data, kp2, final_verified_dmatches, None, 
                                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        max_display_width = 1280
        if img_matches_verified.shape[1] > max_display_width:
            scale_factor = max_display_width / img_matches_verified.shape[1]
            width = int(img_matches_verified.shape[1] * scale_factor)
            height = int(img_matches_verified.shape[0] * scale_factor)
            resized_img_matches_verified = cv2.resize(img_matches_verified, (width, height), interpolation=cv2.INTER_AREA)
        else:
            resized_img_matches_verified = img_matches_verified
            
        cv2.imshow(f"Verified Inlier Matches: {filename1} & {filename2} ({num_final_inliers} inliers)", resized_img_matches_verified)
        cv2.waitKey(1000) # Display for 1 second
        cv2.destroyAllWindows()
        visualization_done_geom_verify = True

# --- Summary of Geometric Verification ---
print("\n--- Geometric Verification Summary ---")
if verified_view_pairs:
    for pair, data in verified_view_pairs.items():
        print(f"Pair: {pair}, Final Inliers: {data['num_inliers']}")
    print(f"Total number of successfully verified pairs: {len(verified_view_pairs)}")
else:
    print("No image pairs were successfully verified.")
