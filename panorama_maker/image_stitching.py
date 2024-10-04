import cv2
import os
import glob
import re
import yaml
import torch
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
    sorted_image_paths = [filepath for _, filepath in image_files_with_timestamps]
    print("Sorted Image Paths:", sorted_image_paths)
    return [filepath for _, filepath in image_files_with_timestamps]


def get_depth_paths(config):
    """Assume depth maps are stored in a separate directory or follow a naming pattern."""
    depth_directory = config["depth_directory"]  # Directory where depth maps are stored
    depth_paths = glob.glob(os.path.join(depth_directory, '*.tiff'))  # Adjust for your file format
    depth_paths.sort()  # Ensure sorting aligns with RGB image paths
    print(f"Found {len(depth_paths)} depth maps.")
    return depth_paths

def extract_timestamp(filename):
    """Extract the timestamp from the filename assuming it is a number before the file extension."""
    basename = os.path.basename(filename)
    # Use regex to find the last sequence of digits before the file extension
    match = re.search(r'(\d+)(?=\.\w+$)', basename)
    if match:
        return int(match.group(1))
    return None

def process_image(path, depth_path, config):
    """Load, crop, and extract features from an image and its depth map without resizing."""
    image_cv = cv2.imread(path)
    print(f"Processing image: {path}")
    if image_cv is None:
        print(f"Error loading image {path}")
        return None, None, None

    # Read the depth image (32-bit TIFF)
    depth_cv = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_cv is None:
        print(f"Error loading depth map {depth_path}")
        return None, None, None

    # Crop both image and depth map
    h, w = image_cv.shape[:2]
    image_cropped = image_cv[config["top_crop"]:h - config["bottom_crop"], config["left_crop"]:w - config["right_crop"]]
    depth_cropped = depth_cv[config["top_crop"]:h - config["bottom_crop"], config["left_crop"]:w - config["right_crop"]]

    # Update size after cropping
    h_cropped, w_cropped = image_cropped.shape[:2]

    # Convert cropped image to tensor without resizing
    image_tensor = transforms.ToTensor()(image_cropped).to(config["device"]).unsqueeze(0)

    # Extract features without resizing
    if config['extractor'] == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(config["device"])
    if config['extractor'] == 'aliked':
        extractor = ALIKED(max_num_keypoints=2048).eval().to(config["device"])

    with torch.no_grad():
        feats = extractor.extract(image_tensor)
        feats['keypoints'] = feats['keypoints'].unsqueeze(0)
        feats['descriptors'] = feats['descriptors'].unsqueeze(0)
        feats['scores'] = feats.get('scores', torch.ones((1, feats['keypoints'].shape[1]), device=feats['keypoints'].device))
        feats = rbd(feats)

    print(f"Extracted features from image: {path}")
    return image_tensor, (h_cropped, w_cropped), feats, depth_cropped  # Return original depth_cropped



def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a NumPy image."""
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image
 # Make sure you have OpenCV installed

def estimateTransformation(srcPoints, dstPoints, srcDepths=None, dstDepths=None, mode='translation_2d'):
    """
    Estimates a transformation matrix based on the given mode and optional depth scaling.
    
    Parameters:
    - srcPoints: Source points (Nx2 array).
    - dstPoints: Destination points (Nx2 array).
    - srcDepths: Depth values at source points (Nx1 array), optional.
    - dstDepths: Depth values at destination points (Nx1 array), optional.
    - mode: The transformation type (translation_2d, rotation_translation, affine_full, etc.).
    
    Modes:
    - 'translation_x': Translation only along the x-axis
    - 'translation_y': Translation only along the y-axis.
    - 'translation_2d': Translation in both x and y directions.
    - 'translation_scale': Translation and uniform scaling.
    - 'rotation_translation': Rotation and translation.
    - 'affine_partial': Equivalent to OpenCV's estimateAffinePartial2D.
    - 'affine_full': Equivalent to OpenCV's estimateAffine2D.
    
    Returns:
    - A 2x3 transformation matrix with depth scaling applied on a per-keypoint basis.
    """
    
    def depth_scaling(depth):
        return np.exp(-0.1 * depth)  # Example scaling function, adjust based on needs

    if srcDepths is not None and dstDepths is not None:
        # Calculate depth scaling factors for each keypoint
        depth_scale_factors = depth_scaling(dstDepths) / depth_scaling(srcDepths)
    else:
        # Default scaling factor if no depth data is provided
        depth_scale_factors = np.ones(srcPoints.shape[0])

    if mode == 'translation_y':
        # Translation along y-axis, apply depth scaling per keypoint
        T_y = 0
        T_x = np.mean((dstPoints[:, 1] - srcPoints[:, 1]) * depth_scale_factors)
        transformation_matrix = np.array([[1, 0, T_x],
                                          [0, 1, T_y]], dtype=np.float32)

    elif mode == 'translation_x':
        # Translation along x-axis, apply depth scaling per keypoint
        T_y = 0
        T_x = np.mean((dstPoints[:, 0] - srcPoints[:, 0]) * depth_scale_factors)
        transformation_matrix = np.array([[1, 0, T_x],
                                          [0, 1, T_y]], dtype=np.float32)

    elif mode == 'translation_2d':
        # Translation in both x and y, apply depth scaling per keypoint
        T_x = np.mean((dstPoints[:, 0] - srcPoints[:, 0]) * depth_scale_factors)
        T_y = np.mean((dstPoints[:, 1] - srcPoints[:, 1]) * depth_scale_factors)
        transformation_matrix = np.array([[1, 0, T_x],
                                          [0, 1, T_y]], dtype=np.float32)

    elif mode == 'translation_scale':
        # Translation + scaling, apply depth scaling per keypoint
        T_x = np.mean((dstPoints[:, 0] - srcPoints[:, 0]) * depth_scale_factors)
        T_y = np.mean((dstPoints[:, 1] - srcPoints[:, 1]) * depth_scale_factors)

        # Calculate scaling factor based on distances and depth scaling per point
        src_distances = np.linalg.norm(srcPoints - np.mean(srcPoints, axis=0), axis=1)
        dst_distances = np.linalg.norm(dstPoints - np.mean(dstPoints, axis=0), axis=1)
        S = np.mean((dst_distances / src_distances) * depth_scale_factors)
        
        transformation_matrix = np.array([[S, 0, T_x],
                                          [0, S, T_y]], dtype=np.float32)

    elif mode == 'rotation_translation':
        # Compute rotation + translation with depth scaling per keypoint
        src_center = np.mean(srcPoints, axis=0)
        dst_center = np.mean(dstPoints, axis=0)
        src_points_centered = srcPoints - src_center
        dst_points_centered = dstPoints - dst_center

        # Compute rotation matrix using SVD
        U, _, Vt = np.linalg.svd(np.dot(dst_points_centered.T, src_points_centered))
        R = np.dot(U, Vt)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(U, Vt)

        # Apply depth scaling per point
        T_x = np.mean((dst_center[0] - np.dot(src_center, R)[0]) * depth_scale_factors)
        T_y = np.mean((dst_center[1] - np.dot(src_center, R)[1]) * depth_scale_factors)
        
        transformation_matrix = np.array([[R[0, 0], R[0, 1], T_x],
                                          [R[1, 0], R[1, 1], T_y]], dtype=np.float32)

    elif mode == 'affine_partial':
        # Custom implementation for depth-aware affine_partial transform
        src_scaled = srcPoints * depth_scale_factors[:, None]
        dst_scaled = dstPoints * depth_scale_factors[:, None]

        # Compute affine transformation using a least-squares approach
        transformation_matrix, _ = cv2.estimateAffinePartial2D(src_scaled, dst_scaled)
        if transformation_matrix is None:
            raise ValueError("Could not estimate affine partial transformation with depth scaling")

    elif mode == 'affine_full':
        # Custom implementation for depth-aware affine_full transform
        src_scaled = srcPoints * depth_scale_factors[:, None]
        dst_scaled = dstPoints * depth_scale_factors[:, None]

        # Compute affine transformation using a least-squares approach
        transformation_matrix, _ = cv2.estimateAffine2D(src_scaled, dst_scaled)
        if transformation_matrix is None:
            raise ValueError("Could not estimate full affine transformation with depth scaling")

    else:
        raise ValueError("Unsupported transformation mode: " + mode)

    return transformation_matrix


def match_keypoints(feats_list, depth_maps, original_sizes, config):
    """Match keypoints between consecutive images and estimate depth-aware transformations."""
    print(f"Matching keypoints for {len(feats_list)} images...")
    
    # Define matching network
    matcher = LightGlue(features=config['extractor']).eval().to(config["device"])

    accumulated_H = np.eye(3)
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
        match_confidences = matches01_rbd["matching_scores0"]

        # Filter valid matches
        valid_matches = (matches > -1) # & (match_confidences > 0.1)
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

        # Adjust keypoints for the cropping
        crop_top = config["top_crop"]
        crop_left = config["left_crop"]

        # Get corresponding depth values for keypoints from depth maps
        depth_map0 = depth_maps[i - 1]
        depth_map1 = depth_maps[i]

        m_kpts0_orig[:, 0] = torch.clamp(m_kpts0_orig[:, 0] - crop_left, 0, depth_map0.shape[1] - 1)
        m_kpts0_orig[:, 1] = torch.clamp(m_kpts0_orig[:, 1] - crop_top, 0, depth_map0.shape[0] - 1)
        m_kpts1_orig[:, 0] = torch.clamp(m_kpts1_orig[:, 0] - crop_left, 0, depth_map1.shape[1] - 1)
        m_kpts1_orig[:, 1] = torch.clamp(m_kpts1_orig[:, 1] - crop_top, 0, depth_map1.shape[0] - 1)

        m_kpts0_np = m_kpts0_orig.cpu().numpy()
        m_kpts1_np = m_kpts1_orig.cpu().numpy()

        # Extract depth values at the keypoints
        src_depths = depth_map0[m_kpts0_orig[:, 1].long(), m_kpts0_orig[:, 0].long()]
        dst_depths = depth_map1[m_kpts1_orig[:, 1].long(), m_kpts1_orig[:, 0].long()]

        if len(m_kpts0_np) >= 3:
            # Estimate depth-aware transformation
            H_affine = estimateTransformation(m_kpts1_np, m_kpts0_np, src_depths, dst_depths, mode=config['pts_transformation'])

            if H_affine is not None:
                H = np.vstack([H_affine, [0, 0, 1]])  # Convert to 3x3 matrix for accumulation
                accumulated_H = accumulated_H @ H  # Update accumulated transformation
                transformations.append(accumulated_H.copy())

                # Update corner transformations
                hi, wi = original_sizes[i]
                corners_i = np.array([[0, 0], [0, hi], [wi, hi], [wi, 0]], dtype=np.float32)
                rotation_translation = accumulated_H[:2, :]
                corners_i_transformed = cv2.transform(corners_i.reshape(-1, 1, 2), rotation_translation)
                all_corners.append(corners_i_transformed.reshape(-1, 2))
                print(f"Transformation found between image {i - 1} and image {i}.")
            else:
                print(f"Transformation failed between image {i-1} and image {i}.")
        else:
            print(f"Not enough matches between image {i-1} and image {i}.")

    return transformations, all_corners


def stitch_images(original_images, transformations, all_corners):
    """Stitch images together into a panorama using original blending logic."""
    print("Starting stitching of images...")
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
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    mask_panorama = np.zeros((panorama_height, panorama_width), dtype=np.uint8)

    # Warp and blend each image using the original blending logic
    for i in range(len(original_images)):
        stitch_count += 1
        print(f"Stitching image {stitch_count}/{total_stitching}")

        H = transformations[i]
        # Compute the total transformation
        H_total = translation @ H

        # Warp the image
        image_i_np = tensor_to_image(original_images[i])
        warped_image_i = cv2.warpPerspective(image_i_np, H_total, (panorama_width, panorama_height))

        # Warp the mask
        hi, wi = image_i_np.shape[:2]
        mask_i = np.ones((hi, wi), dtype=np.uint8) * 255
        warped_mask_i = cv2.warpPerspective(mask_i, H_total, (panorama_width, panorama_height))

        # Update panorama and mask_panorama with original blending logic
        mask_overlap = warped_mask_i > 0
        panorama[mask_overlap] = warped_image_i[mask_overlap]
        mask_panorama[mask_overlap] = warped_mask_i[mask_overlap]
        print(f"Stitching image {i + 1} of {len(original_images)}")

    print("Stitching completed.")
    return panorama

def run_stitching_pipeline(config_path):
    """Run the complete stitching pipeline with configuration from a YAML file."""
    print("Running image stitching pipeline...")

    # Load the configuration from YAML file
    config = load_config(config_path)

    # Get paths for RGB images
    image_paths = get_image_paths(config)

    # Assuming depth maps are stored in a similar directory or naming pattern as the image files
    depth_paths = get_depth_paths(config)  # Implement get_depth_paths to load depth map paths

    # Check if the number of image and depth map files match
    assert len(image_paths) == len(depth_paths), "Mismatch between image files and depth maps."

    print("Processing images and depth maps...")
    total_images = len(image_paths)

    original_images = []
    original_sizes = []
    feats_list = []
    depth_maps = []

    # Use ThreadPoolExecutor for parallel processing of images
    with ThreadPoolExecutor() as executor:
        # Pair image paths with their indices
        futures = {
            executor.submit(process_image, image_paths[i], depth_paths[i], config): i
            for i in range(len(image_paths))
        }

        # Process the results, ensuring the order is preserved
        results = []
        for future in futures:
            index = futures[future]
            image_tensor, size, feats, depth_resized = future.result()  # Get the result
            if image_tensor is not None:
                results.append((index, image_tensor, size, feats, depth_resized))

        # Sort the results based on the original index to ensure the order
        results.sort(key=lambda x: x[0])

    # Extract sorted data
    for _, image_tensor, size, feats, depth_resized in results:
        original_images.append(image_tensor)
        original_sizes.append(size)
        feats_list.append(feats)
        depth_maps.append(depth_resized)

    # Match keypoints using depth maps and estimate transformations
    print("Matching keypoints and estimating transformations...")
    transformations, all_corners = match_keypoints(feats_list, depth_maps, original_sizes, config)

    # Stitch images together using the computed transformations
    print("Stitching images...")
    panorama = stitch_images(original_images, transformations, all_corners)

    # Save the panorama image if requested
    if config['save_image']:
        cv2.imwrite(os.path.join(config['output_dir'], config['output_filename']), panorama)
        print(f"Panorama saved at {os.path.join(config['output_dir'], config['output_filename'])}")

    return panorama


# Use this guard to prevent auto-execution when imported
if __name__ == "__main__":
    config_path = "config.yaml"
    panorama = run_stitching_pipeline(config_path)