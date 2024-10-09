import cv2
import os
import glob
import re
import yaml
import torch
import numba
import time
from numba import jit, njit
import numpy as np
from lightglue import LightGlue, SuperPoint, ALIKED
from lightglue.utils import rbd
from torchvision import transforms

from concurrent.futures import ThreadPoolExecutor

def load_config(config_path):
    """Load configuration from a YAML file and compile regex patterns."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Adjust device setting based on the string from config
    config["device"] = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    print("Configuration loaded successfully.")
    return config

def get_image_paths(config):
    """Get sorted image paths based on timestamps from filenames."""
    all_image_files = glob.glob(os.path.join(config["image_directory"], '*.*'))  # Adjust the pattern to include all files
    image_files_with_timestamps = []

    for filepath in all_image_files:
        timestamp = extract_timestamp(filepath)
        if timestamp is not None:
            image_files_with_timestamps.append((timestamp, filepath))
        else:
            print(f"Filename {filepath} does not contain a valid timestamp and will be skipped.")

    # Sort the image files based on timestamp
    image_files_with_timestamps.sort(key=lambda x: x[0])
    print(f"Found {len(image_files_with_timestamps)} images to process.")
    return [filepath for _, filepath in image_files_with_timestamps]

    # Sort the image files based on timestamp
    image_files_with_timestamps.sort(key=lambda x: x[0])
    return [filepath for _, filepath in image_files_with_timestamps]

def extract_timestamp(filename):
    """Extract the timestamp from the filename assuming it is a number before the file extension."""
    basename = os.path.basename(filename)
    # Use regex to find the last sequence of digits before the file extension
    match = re.search(r'(\d+)(?=\.\w+$)', basename)
    if match:
        return int(match.group(1))
    return None

def process_image(path, config):
    """Load, crop, resize, and extract features from an image."""
    image_cv = cv2.imread(path)
    print(f"Processing image: {path}")
    if image_cv is None:
        print(f"Error loading image {path}")
        return None, None, None

    # Crop image
    h, w = image_cv.shape[:2]
    image_cropped = image_cv[config["top_crop"]:h - config["bottom_crop"], config["left_crop"]:w - config["right_crop"]]

    # Update size after cropping
    h_cropped, w_cropped = image_cropped.shape[:2]

    # Convert cropped image to tensor
    image_tensor = transforms.ToTensor()(image_cropped).to(config["device"]).unsqueeze(0)

    # Resize image for feature extraction
    image_resized = cv2.resize(image_cropped, (config["fixed_width"], config["fixed_height"]), interpolation=cv2.INTER_LINEAR)
    image_resized_tensor = transforms.ToTensor()(image_resized).to(config["device"]).unsqueeze(0)

    # Extract features
    if config['extractor'] == 'superpoint':
        print('Using SuperPoint for feature extraction')
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(config["device"])
    if config['extractor'] == 'aliked':
        print('Using ALIKED for feature extraction')
        extractor = ALIKED(max_num_keypoints=2048).eval().to(config["device"])

    with torch.no_grad():
        feats = extractor.extract(image_resized_tensor)
        feats['keypoints'] = feats['keypoints'].unsqueeze(0)
        feats['descriptors'] = feats['descriptors'].unsqueeze(0)
        feats['scores'] = feats.get('scores', torch.ones((1, feats['keypoints'].shape[1]), device=feats['keypoints'].device))
        feats = rbd(feats)

    print(f"Extracted features from image: {path}")
    return image_tensor, (h_cropped, w_cropped), feats

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a NumPy image."""
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image

import numpy as np
import cv2  # Make sure you have OpenCV installed

import cv2
import numpy as np

def estimateTransformation(srcPoints, dstPoints, mode='translation_2d'):
    """
    Estimates a transformation matrix based on the given mode.
    
    Modes:
    - 'translation_x': Translation only along the x-axis
    - 'translation_y': Translation only along the y-axis.
    - 'translation_2d': Translation in both x and y directions.
    - 'translation_scale': Translation and uniform scaling.
    - 'rotation_translation': Rotation and translation.
    - 'affine_partial': Equivalent to OpenCV's estimateAffinePartial2D.
    - 'affine_full': Equivalent to OpenCV's estimateAffine2D.
    - 'homography': Full homography transformation.
    
    Parameters:
    - srcPoints: Source points (Nx2 array).
    - dstPoints: Destination points (Nx2 array).
    - mode: The transformation type.
    
    Returns:
    - A 3x3 transformation matrix.
    """
    
    if mode == 'translation_y':
        T_x = 0
        T_y = np.mean(dstPoints[:, 1] - srcPoints[:, 1])
        # Create 3x3 homogeneous matrix for translation along y
        transformation_matrix = np.array([[1, 0, T_x],
                                          [0, 1, T_y],
                                          [0, 0, 1]], dtype=np.float32)
    
    elif mode == 'translation_x':
        T_x = np.mean(dstPoints[:, 0] - srcPoints[:, 0])
        T_y = 0
        # Create 3x3 homogeneous matrix for translation along x
        transformation_matrix = np.array([[1, 0, T_x],
                                          [0, 1, T_y],
                                          [0, 0, 1]], dtype=np.float32)
    
    elif mode == 'translation_2d':
        T_x = np.mean(dstPoints[:, 0] - srcPoints[:, 0])
        T_y = np.mean(dstPoints[:, 1] - srcPoints[:, 1])
        # Create 3x3 homogeneous matrix for 2D translation
        transformation_matrix = np.array([[1, 0, T_x],
                                          [0, 1, T_y],
                                          [0, 0, 1]], dtype=np.float32)
    
    elif mode == 'translation_scale':
        T_x = np.mean(dstPoints[:, 0] - srcPoints[:, 0])
        T_y = np.mean(dstPoints[:, 1] - srcPoints[:, 1])
        
        src_distances = np.linalg.norm(srcPoints - np.mean(srcPoints, axis=0), axis=1)
        dst_distances = np.linalg.norm(dstPoints - np.mean(dstPoints, axis=0), axis=1)
        S = np.mean(dst_distances / src_distances)
        
        transformation_matrix = np.array([[S, 0, T_x],
                                          [0, S, T_y],
                                          [0, 0, 1]], dtype=np.float32)
    
    elif mode == 'rotation_translation':
        src_center = np.mean(srcPoints, axis=0)
        dst_center = np.mean(dstPoints, axis=0)

        src_points_centered = srcPoints - src_center
        dst_points_centered = dstPoints - dst_center

        U, _, Vt = np.linalg.svd(np.dot(dst_points_centered.T, src_points_centered))
        R = np.dot(U, Vt)

        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(U, Vt)

        T_x = dst_center[0] - np.dot(src_center, R)[0]
        T_y = dst_center[1] - np.dot(src_center, R)[1]

        transformation_matrix = np.array([[R[0, 0], R[0, 1], T_x],
                                          [R[1, 0], R[1, 1], T_y],
                                          [0, 0, 1]], dtype=np.float32)
    
    elif mode == 'affine_partial':
        transformation_matrix, _ = cv2.estimateAffinePartial2D(srcPoints, dstPoints)
        if transformation_matrix is None:
            raise ValueError("Could not estimate affine partial transformation")
        # Convert to 3x3 matrix
        transformation_matrix = np.vstack([transformation_matrix, [0, 0, 1]])
    
    elif mode == 'affine_full':
        transformation_matrix, _ = cv2.estimateAffine2D(srcPoints, dstPoints)
        if transformation_matrix is None:
            raise ValueError("Could not estimate full affine transformation")
        # Convert to 3x3 matrix
        transformation_matrix = np.vstack([transformation_matrix, [0, 0, 1]])
    
    elif mode == 'homography':
        transformation_matrix, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC)
        if transformation_matrix is None:
            raise ValueError("Could not estimate homography transformation")
    
    else:
        raise ValueError("Unknown mode. Supported modes are 'translation_y', 'translation_x', 'translation_2d', 'translation_scale', 'rotation_translation', 'affine_partial', 'affine_full', 'homography'.")
    
    return transformation_matrix


def match_keypoints(feats_list, original_sizes, config):
    """Match keypoints between consecutive images and estimate transformations."""
    print(f"Matching keypoints for {len(feats_list)} images...")

    # Define matching network
    matcher = LightGlue(features=config['extractor']).eval().to(config["device"])

    # Initialize accumulated transformation matrix as 3x3 identity matrix
    accumulated_H = np.eye(3, dtype=np.float32)
    transformations = [accumulated_H.copy()]
    all_corners = []

    total_matches = len(feats_list) - 1
    match_count = 0

    for i in range(1, len(feats_list)):
        match_count += 1
        print(f"Matching keypoints between image {i - 1} and image {i} ({match_count}/{total_matches})")

        feats0, feats1 = feats_list[i - 1], feats_list[i]
        matches_input = {"image0": feats0, "image1": feats1}
        matches01 = matcher(matches_input)
        matches01_rbd = rbd(matches01)

        kpts0 = feats0["keypoints"].squeeze(0)
        kpts1 = feats1["keypoints"].squeeze(0)
        matches = matches01_rbd["matches0"].squeeze(0)

        # Filter valid matches
        valid_matches = matches > -1
        m_kpts0 = kpts0[valid_matches]
        m_kpts1 = kpts1[matches[valid_matches]]

        # Scale keypoints back to original image dimensions
        h_resized, w_resized = config["fixed_height"], config["fixed_width"]
        h0_orig, w0_orig = original_sizes[i - 1]
        hi_orig, wi_orig = original_sizes[i]

        scale_x0 = w0_orig / w_resized
        scale_y0 = h0_orig / h_resized
        scale_xi = wi_orig / w_resized
        scale_yi = hi_orig / h_resized

        m_kpts0_orig = m_kpts0.clone()
        m_kpts0_orig[:, 0] *= scale_x0
        m_kpts0_orig[:, 1] *= scale_y0

        m_kpts1_orig = m_kpts1.clone()
        m_kpts1_orig[:, 0] *= scale_xi
        m_kpts1_orig[:, 1] *= scale_yi

        m_kpts0_np = m_kpts0_orig.cpu().numpy()
        m_kpts1_np = m_kpts1_orig.cpu().numpy()

        if len(m_kpts0_np) >= 3:
            # Estimate the transformation matrix based on the configuration mode
            H_estimated = estimateTransformation(m_kpts1_np, m_kpts0_np, mode=config['pts_transformation'])

            if H_estimated is not None:
                # Convert to 3x3 if it's a 2x3 affine transformation
                if H_estimated.shape == (2, 3):
                    H = np.vstack([H_estimated, [0, 0, 1]])
                else:
                    H = H_estimated  # For homography or other 3x3 matrices

                # Accumulate the transformation
                accumulated_H = accumulated_H @ H
                transformations.append(accumulated_H.copy())

                # Update corners for the transformed image
                hi, wi = original_sizes[i]
                corners_i = np.array([[0, 0], [0, hi], [wi, hi], [wi, 0]], dtype=np.float32)
                corners_i_transformed = cv2.perspectiveTransform(corners_i.reshape(-1, 1, 2), accumulated_H)
                all_corners.append(corners_i_transformed.reshape(-1, 2))

                print(f"Transformation found between image {i - 1} and image {i}.")
            else:
                print(f"Transformation failed between image {i - 1} and image {i}.")
        else:
            print(f"Not enough matches between image {i - 1} and image {i}.")

    return transformations, all_corners

import matplotlib.pyplot as plt
# Cache for gradient masks of different sizes
# Cache for gradient masks of different sizes
gradient_mask_cache = {}

def warp_partial(image, H, output_shape, min_x, min_y, max_x, max_y):
    """Warp only a portion of the image defined by a bounding box, with detailed profiling."""
    height, width = output_shape[:2]

    # Start profiling
    start_time = time.time()
    
    # Create bounding box with homogeneous coordinates
    bbox = np.array([[min_x, min_y, 1], [max_x, min_y, 1], 
                     [max_x, max_y, 1], [min_x, max_y, 1]], dtype=np.float32).T
    bbox_time = time.time()
    print(f"Bounding box creation time: {bbox_time - start_time:.4f} seconds")
    
    # Apply transformation matrix to the bounding box points
    bbox_transformed = H @ bbox
    bbox_transformed /= bbox_transformed[2]  # Convert back to 2D by dividing by the last coordinate
    transform_time = time.time()
    print(f"Transformation application time: {transform_time - bbox_time:.4f} seconds")

    # Extract transformed points for the new ROI bounding box
    transformed_bbox = bbox_transformed[:2].T.astype(np.float32).reshape(4, 2)
    extraction_time = time.time()
    print(f"Bounding box extraction time: {extraction_time - transform_time:.4f} seconds")

    # Original bbox points (source) for getPerspectiveTransform
    bbox_points = np.array([[min_x, min_y], [max_x, min_y], 
                            [max_x, max_y], [min_x, max_y]], dtype=np.float32)
    perspective_time = time.time()
    print(f"Perspective points setup time: {perspective_time - extraction_time:.4f} seconds")

    # Calculate the perspective transform for just the ROI
    H_roi = cv2.getPerspectiveTransform(bbox_points, transformed_bbox)
    perspective_transform_time = time.time()
    print(f"Perspective transformation calculation time: {perspective_transform_time - perspective_time:.4f} seconds")

    # Warp only the selected ROI
    roi_warped = cv2.warpPerspective(image, H_roi, (width, height), flags=cv2.INTER_NEAREST)
    warp_end_time = time.time()
    print(f"Warp operation time: {warp_end_time - perspective_transform_time:.4f} seconds")

    total_time = warp_end_time - start_time
    print(f"Total warp_partial execution time: {total_time:.4f} seconds")
    
    return roi_warped


# def generate_gradient_for_mask(mask, blending_buffer_width):
#     """Generate a gradient mask specific to the warped mask's content."""
#     # Identify the bounding box of the masked region to limit the gradient area
#     coords = np.column_stack(np.where(mask > 0))
#     y_min, x_min = coords.min(axis=0)
#     y_max, x_max = coords.max(axis=0)

#     # Define the shape of the masked region
#     mask_height, mask_width = y_max - y_min + 1, x_max - x_min + 1
    
#     # Create a radial gradient mask for the masked area only
#     mask_shape = (mask_height, mask_width)
#     return create_radial_gradient_mask(mask_shape, blending_buffer_width), (y_min, y_max, x_min, x_max)

# @njit
# def create_radial_gradient_mask(shape, buffer_width):
#     """Creates a gradient mask within a specified area that falls off from 1 at the edges to 0."""
#     h, w = shape
#     gradient_mask = np.zeros((h, w), dtype=np.float32)
#     for y in range(h):
#         for x in range(w):
#             dist_to_edge = min(x, w - x - 1, y, h - y - 1)  # Distance to the nearest edge
#             gradient_value = max(0, min(1, dist_to_edge / buffer_width))
#             gradient_mask[y, x] = gradient_value
#     return gradient_mask


# @njit
# def create_radial_gradient_mask(shape, buffer_width):
#     """Creates a gradient mask that falls off from 1 at the edge to 0 over the buffer distance towards the center."""
#     h, w = shape
#     gradient_mask = np.zeros((h, w), dtype=np.float32)

#     for y in range(h):
#         for x in range(w):
#             dist_to_edge = min(x, w - x - 1, y, h - y - 1)  # Distance to the nearest edge
#             gradient_value = max(0, min(1, dist_to_edge / buffer_width))
#             gradient_mask[y, x] = gradient_value

#     return gradient_mask

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import zoom


# Cache dictionary for storing gradient masks
# Cache dictionary for storing gradient masks
gradient_cache = {}

import time

def create_radial_gradient_mask_approx(shape, buffer_width):
    """Creates an approximate radial gradient mask using pure numpy vectorization with detailed profiling."""
    h, w = shape
    
    # Start profiling
    start_time = time.time()
    
    # Create grids for x and y coordinates
    y_indices, x_indices = np.ogrid[:h, :w]
    coord_grid_time = time.time()
    print(f"Coordinate grid creation time: {coord_grid_time - start_time:.4f} seconds")
    
    # Calculate distances to each edge
    dist_to_left = x_indices
    dist_to_right = w - x_indices - 1
    dist_to_top = y_indices
    dist_to_bottom = h - y_indices - 1
    edge_distance_time = time.time()
    print(f"Edge distance calculation time: {edge_distance_time - coord_grid_time:.4f} seconds")
    
    # Combine distances using chained np.minimum calls
    dist_to_edge = np.minimum(dist_to_left, dist_to_right)
    dist_to_edge = np.minimum(dist_to_edge, dist_to_top)
    dist_to_edge = np.minimum(dist_to_edge, dist_to_bottom)
    min_distance_time = time.time()
    print(f"Minimum distance calculation time: {min_distance_time - edge_distance_time:.4f} seconds")
    
    # Normalize and clip based on buffer width
    gradient_mask = np.clip(dist_to_edge / buffer_width, 0, 1).astype(np.float32)
    normalize_time = time.time()
    print(f"Normalization and clipping time: {normalize_time - min_distance_time:.4f} seconds")
    
    # Total time
    total_time = normalize_time - start_time
    print(f"Total create_radial_gradient_mask_approx time: {total_time:.4f} seconds")
    
    return gradient_mask

def optimized_boundary_detection(mask):
    """Optimized boundary detection with purely numpy operations for speed."""
    # Find non-zero rows and columns
    non_zero_rows = np.where(mask.any(axis=1))[0]
    non_zero_cols = np.where(mask.any(axis=0))[0]
    # Get boundaries
    y_min, y_max = non_zero_rows[[0, -1]]
    x_min, x_max = non_zero_cols[[0, -1]]
    return y_min, y_max, x_min, x_max


# Cache dictionary for storing gradient masks
gradient_cache = {}

def generate_gradient_for_mask(mask, blending_buffer_width):
    """Generate a gradient mask specific to the warped mask's content, with caching for performance."""
    start_time = time.time()

    # Finding mask boundaries
    coords_start = time.time()
    y_min, y_max, x_min, x_max = optimized_boundary_detection(mask)
    mask_shape = (y_max - y_min + 1, x_max - x_min + 1)
    coords_end = time.time()
    print(f"Coords finding time: {coords_end - coords_start:.4f} seconds")

    # Caching
    cache_start = time.time()
    cache_key = (mask_shape, blending_buffer_width)
    if cache_key in gradient_cache:
        gradient_mask = gradient_cache[cache_key]
        print(f"Cache hit for shape {mask_shape}")
    else:
        # Generate gradient mask and cache it
        gradient_mask = create_radial_gradient_mask_approx(mask_shape, blending_buffer_width)
        gradient_cache[cache_key] = gradient_mask
        print(f"Cache miss for shape {mask_shape}")
    cache_end = time.time()
    print(f"Cache check and retrieval time: {cache_end - cache_start:.4f} seconds")

    total_time = time.time() - start_time
    print(f"Total execution time for generate_gradient_for_mask: {total_time:.4f} seconds")

    return gradient_mask, (y_min, y_max, x_min, x_max)


import time

@njit
def optimized_blend_regions(panorama, warped_image, weight_map, mask_overlap, mask_non_overlap):
    """Efficiently blend overlapping and non-overlapping regions with Numba."""
    h, w = mask_overlap.shape
    for y in range(h):
        for x in range(w):
            if mask_overlap[y, x]:
                for c in range(3):  # For each color channel
                    panorama[y, x, c] = (
                        weight_map[y, x] * warped_image[y, x, c] +
                        (1 - weight_map[y, x]) * panorama[y, x, c]
                    )
            elif mask_non_overlap[y, x]:
                # Directly copy non-overlapping pixels from warped image
                panorama[y, x] = warped_image[y, x]
    return panorama


import time

def stitch_images(original_images, transformations, all_corners, blending_buffer_width=50):
    """Stitch images with blending specifically within the warped mask area, with time profiling."""
    all_corners = np.vstack(all_corners)
    x_min, y_min = np.int32(all_corners.min(axis=0) - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0) + 0.5)
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min

    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    panorama = np.full((panorama_height, panorama_width, 3), -1, dtype=np.float32)
    mask_panorama = np.zeros((panorama_height, panorama_width), dtype=np.uint8)

    start_stitch_time = time.time()
    for i, image_tensor in enumerate(original_images):
        print(f"\nStitching image {i + 1}/{len(original_images)}")
        start_time_stitch_i = time.time()

        # Transformation application
        transform_start = time.time()
        H_total = translation @ transformations[i]
        transform_end = time.time()
        print(f"Transformation time: {transform_end - transform_start:.4f} seconds")

        # Warping image and mask
        warp_start = time.time()
        image_np = tensor_to_image(image_tensor)
        warped_image = warp_partial(image_np, H_total, (panorama_height, panorama_width), x_min, y_min, x_max, y_max)
        warped_mask = warp_partial(np.ones(image_np.shape[:2], dtype=np.uint8) * 255, H_total, (panorama_height, panorama_width), x_min, y_min, x_max, y_max)
        warp_end = time.time()
        print(f"Warping time: {warp_end - warp_start:.4f} seconds")

        # Generating the gradient mask for the warped mask content
        gradient_start = time.time()
        gradient_mask, (y_min_g, y_max_g, x_min_g, x_max_g) = generate_gradient_for_mask(warped_mask, blending_buffer_width)
        gradient_end = time.time()
        print(f"Gradient mask generation time: {gradient_end - gradient_start:.4f} seconds")

        # Calculating overlap and applying blending
        overlap_start = time.time()
        mask_overlap = (mask_panorama[y_min_g:y_max_g+1, x_min_g:x_max_g+1] > 0) & (warped_mask[y_min_g:y_max_g+1, x_min_g:x_max_g+1] > 0)
        mask_non_overlap = (mask_panorama[y_min_g:y_max_g+1, x_min_g:x_max_g+1] == 0) & (warped_mask[y_min_g:y_max_g+1, x_min_g:x_max_g+1] > 0)
        overlap_end = time.time()
        print(f"Overlap calculation time: {overlap_end - overlap_start:.4f} seconds")

        blending_start = time.time()
        effective_gradient = np.where(mask_overlap, gradient_mask, 1)
        panorama[y_min_g:y_max_g+1, x_min_g:x_max_g+1] = optimized_blend_regions(
            panorama[y_min_g:y_max_g+1, x_min_g:x_max_g+1],
            warped_image[y_min_g:y_max_g+1, x_min_g:x_max_g+1],
            effective_gradient,
            mask_overlap,
            mask_non_overlap
        )
        blending_end = time.time()
        print(f"Blending time: {blending_end - blending_start:.4f} seconds")

        # Update the panorama mask
        mask_update_start = time.time()
        # mask_panorama[warped_mask > 0] = 255
        np.bitwise_or(mask_panorama, warped_mask, out=mask_panorama)
        mask_update_end = time.time()
        print(f"Mask update time: {mask_update_end - mask_update_start:.4f} seconds")

        end_time_stitch_i = time.time()
        print(f"Total time to stitch image: {end_time_stitch_i - start_time_stitch_i:.4f} seconds")

    # Finalize the panorama
    finalize_start = time.time()
    panorama[panorama == -1] = 0  # Replace unassigned areas with black
    panorama = panorama.astype(np.uint8)
    finalize_end = time.time()
    print(f"Finalization time: {finalize_end - finalize_start:.4f} seconds")

    total_time = finalize_end - start_stitch_time
    print(f"Total stitching time: {total_time:.4f} seconds")

    return panorama


def run_stitching_pipeline(config_path, save_img=False):
    """Run the complete stitching pipeline with configuration from a YAML file."""
    print("Running image stitching pipeline...")
    config = load_config(config_path)
    image_paths = get_image_paths(config)

    print("Processing images...")
    total_images = len(image_paths)
    processed_count = 0

    original_images = []
    original_sizes = []
    feats_list = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda path: process_image(path, config), image_paths)

    for image_tensor, size, feats in results:
        processed_count += 1
        print(f"Processed {processed_count}/{total_images} images.")
        if image_tensor is not None:
            original_images.append(image_tensor)
            original_sizes.append(size)
            feats_list.append(feats)

    print("Matching keypoints and estimating transformations...")
    transformations, all_corners = match_keypoints(feats_list, original_sizes, config)
    panorama = stitch_images(original_images, transformations, all_corners)
    print("Pipeline completed.")

    if config['save_image']:
        cv2.imwrite(os.path.join(config['output_dir'], config['output_filename']), panorama)

    return panorama

# Use this guard to prevent auto-execution when imported
if __name__ == "__main__":
    config_path = "config.yaml"
    panorama = run_stitching_pipeline(config_path)