import cv2
import os
import glob
import re
import yaml
import torch
import numba
from numba import jit
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

import cv2
import numpy as np
from numba import jit

@jit(nopython=True)
def blend_regions(panorama, warped_image, weight_map, mask_overlap, valid_overlap, mask_non_overlap):
    """Blend the overlapping and non-overlapping regions using Numba for speedup."""
    # Smoothly blend overlapping regions using the Gaussian weight map
    for y in range(panorama.shape[0]):
        for x in range(panorama.shape[1]):
            if valid_overlap[y, x]:
                for c in range(3):  # Iterate over each color channel (R, G, B)
                    panorama[y, x, c] = (weight_map[y, x, c] * warped_image[y, x, c] +
                                         (1 - weight_map[y, x, c]) * panorama[y, x, c])
            elif mask_non_overlap[y, x]:
                for c in range(3):
                    panorama[y, x, c] = warped_image[y, x, c]

    return panorama

import time

def apply_box_blur_multiple_passes(image, kernel_size, passes=3):
    """Apply multiple passes of a box filter to approximate Gaussian blur."""
    blurred = image.copy()
    for _ in range(passes):
        blurred = cv2.boxFilter(blurred, -1, (kernel_size, kernel_size))
    return blurred

def stitch_images(original_images, transformations, all_corners):
    """Stitch images together into a panorama using box filter approximation for smoother transitions."""
    print("Starting stitching of images...")
    
    # Start overall timer
    start_time = time.time()
    
    all_corners = np.vstack(all_corners)
    x_min, y_min = np.int32(all_corners.min(axis=0) - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0) + 0.5)
    panorama_width = x_max - x_min
    panorama_height = y_max - y_min

    total_stitching = len(original_images)
    stitch_count = 0

    # Compute the translation matrix
    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])

    # Initialize the panorama and mask
    panorama = np.full((panorama_height, panorama_width, 3), -1, dtype=np.float32)
    mask_panorama = np.zeros((panorama_height, panorama_width), dtype=np.uint8)

    # Baseline kernel size ratio for Gaussian blur (tunable)
    base_kernel_ratio = 0.163  # Derived value for optimal blending based on image size

    # Start processing each image
    for i in range(len(original_images)):
        stitch_count += 1
        print(f"Stitching image {stitch_count}/{total_stitching}")

        # Start time for each image stitching
        image_start_time = time.time()

        H = transformations[i]
        H_total = translation @ H

        # Time warp operation
        warp_start = time.time()
        # Warp the image using nearest neighbor interpolation to avoid new artifacts
        image_i_np = tensor_to_image(original_images[i])
        warped_image_i = cv2.warpPerspective(image_i_np, H_total, (panorama_width, panorama_height), flags=cv2.INTER_NEAREST).astype(np.float32)
        warp_end = time.time()
        print(f"Image {i + 1}: Warp time: {warp_end - warp_start:.4f} seconds")

        # Time mask creation
        mask_start = time.time()
        # Create a mask and warp it with expanded corners for slight overlap
        hi, wi = image_i_np.shape[:2]
        mask_i = np.ones((hi, wi), dtype=np.uint8) * 255

        # Create an expanded version of the mask to ensure slight overlap
        expansion_kernel = np.ones((5, 5), np.uint8)  # Expand by 1-2 pixels using a 5x5 kernel
        expanded_mask_i = cv2.dilate(mask_i, expansion_kernel, iterations=1)

        warped_mask_i = cv2.warpPerspective(expanded_mask_i, H_total, (panorama_width, panorama_height), flags=cv2.INTER_NEAREST).astype(np.float32) / 255
        mask_end = time.time()
        print(f"Image {i + 1}: Mask creation time: {mask_end - mask_start:.4f} seconds")

        # Time box blur approximation
        blur_start = time.time()
        # Identify overlapping and non-overlapping regions
        mask_overlap = (mask_panorama > 0) & (warped_mask_i > 0)
        mask_non_overlap = (mask_panorama == 0) & (warped_mask_i > 0)

        # Calculate overlap percentage
        overlap_area = np.sum(mask_overlap)
        total_area = warped_image_i.shape[0] * warped_image_i.shape[1]
        overlap_percentage = overlap_area / total_area if total_area > 0 else 0

        # Dynamically calculate the box kernel size based on overlap and image dimensions
        min_dim = min(hi, wi)
        dynamic_kernel_ratio = base_kernel_ratio * (1 + overlap_percentage)  # Adjust kernel ratio based on overlap
        kernel_size = max(int(min_dim * dynamic_kernel_ratio), 3)

        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Cap the kernel size to prevent excessively large kernel values
        max_kernel_size = 1001  # This can be adjusted based on your preference
        kernel_size = min(kernel_size, max_kernel_size)

        # Apply box blur with multiple passes to approximate Gaussian blur
        weight_map = apply_box_blur_multiple_passes(warped_mask_i, kernel_size, passes=3)

        blur_end = time.time()
        print(f"Image {i + 1}: Box blur time: {blur_end - blur_start:.4f} seconds")

        # Expand weight map to match RGB channels if the image has 3 channels
        expand_start = time.time()
        if warped_image_i.shape[-1] == 3:
            weight_map = np.stack([weight_map] * 3, axis=-1)
        expand_end = time.time()
        print(f"Image {i + 1}: Expand weight map time: {expand_end - expand_start:.4f} seconds")

        # Update blending to ignore areas initialized with -1
        valid_overlap = mask_overlap & (panorama[..., 0] != -1)

        # Convert boolean masks to integer masks (as numba does not handle boolean indexing well)
        mask_overlap = mask_overlap.astype(np.uint8)
        valid_overlap = valid_overlap.astype(np.uint8)
        mask_non_overlap = mask_non_overlap.astype(np.uint8)

        # Blend overlapping and non-overlapping regions using numba-accelerated function
        blend_start = time.time()
        panorama = blend_regions(panorama, warped_image_i, weight_map, mask_overlap, valid_overlap, mask_non_overlap)
        blend_end = time.time()
        print(f"Image {i + 1}: Blending time: {blend_end - blend_start:.4f} seconds")

        # Update the mask_panorama to mark the new covered regions
        mask_panorama[warped_mask_i > 0] = 255

        # End time for each image stitching
        image_end_time = time.time()
        print(f"Image {i + 1}: Total stitching time: {image_end_time - image_start_time:.4f} seconds")

    # Replace areas initialized with -1 with zeros (or another suitable background value)
    final_replace_start = time.time()
    panorama[panorama == -1] = 0
    panorama = panorama.astype(np.uint8)
    final_replace_end = time.time()
    print(f"Final replace time: {final_replace_end - final_replace_start:.4f} seconds")

    # End overall timer
    end_time = time.time()
    print(f"Stitching completed in {end_time - start_time:.4f} seconds.")
    
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