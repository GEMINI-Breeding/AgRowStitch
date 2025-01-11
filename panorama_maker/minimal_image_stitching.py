# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:33:41 2024

@author: Kaz Uyehara
"""
import cv2
from lightglue import LightGlue, SuperPoint #git clone https://github.com/cvg/LightGlue.git && cd LightGlue
from lightglue.utils import rbd
import itertools
import numpy as np
import os

from stitching.blender import Blender

import sys
import torch
from torchvision import transforms
import yaml


def extract_features(img_path, config):
    ##################################
    #Load image and reduce resolution#
    ##################################
    image = cv2.imread(img_path) #Read image and then resize
    medium_img = cv2.resize(image, dsize = None, fx = config["feature_resolution"], fy = config["feature_resolution"])

    ####################################
    #Use SuperPoint to extract features#
    ####################################
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(config["device"])
    image_resized_tensor = transforms.ToTensor()(medium_img).to(config["device"]).unsqueeze(0)
    
    with torch.no_grad():
        feats = extractor.extract(image_resized_tensor)
        feats['scores'] = feats.get('scores', torch.ones((1, feats['keypoints'].shape[1]), device=feats['keypoints'].device))
        feats = rbd(feats)
        feats['keypoints'] = feats['keypoints'].unsqueeze(0)
        feats['descriptors'] = feats['descriptors'].unsqueeze(0)
        feats['keypoint_scores']= feats['keypoint_scores']
        
    return feats
        

def get_inliers(img_feats, img_paths, src_idx, dst_idx, img_dim, config):
    ###########################################################
    #Use LightGlue to match features extracted from SuperPoint#
    ###########################################################
    #Only extract features if not already present to minimize VRAM use
    if src_idx not in img_feats:
        img_feats[src_idx] = extract_features(img_paths[src_idx], config)
    if dst_idx not in img_feats:
        img_feats[dst_idx] = extract_features(img_paths[dst_idx], config)

    matcher = LightGlue(features="superpoint").eval().to(config["device"])
    feat_dict = {'image0': img_feats[src_idx], 'image1': img_feats[dst_idx]}
    img_matches = matcher(feat_dict)
    feats0, feats1, img_matches = [rbd(x) for x in [feat_dict['image0'], feat_dict['image1'], img_matches]]
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], img_matches["matches"]

    ###################################################
    #Find which keypoints were matched and move to cpu#
    ###################################################
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    m_kpts0_np = m_kpts0.cpu().numpy()
    m_kpts1_np = m_kpts1.cpu().numpy()
    
    ##############################################
    #Subset the matching points based on position#
    ##############################################
    #We prefer the images to have keypoints close to their stitching edge so that keypoints are
    #unlikely to appear more than twice and we can minimize the number of images used.
    #The keypoint prop is the part of the image we expect to find keypoints, e.g. if 
    #keypoint prop is 0.5, we want the keypoints to be on the correct half of both images -- 
    #closer to the stitching edge than the non-stitching edge
    keypoint_prop_dict = {} #assume keys will stay ordered
    
    #We assume that for forward movement a keypoint_prop of at least 0.9 is necessary, so it is the upper limit
    for keypoint_prop in np.arange(config["keypoint_prop"], 1.0, 0.1): 
        if config["stitching_direction"] == 'LEFT' or config["stitching_direction"] == 'UP':
            src_pixel_limit = img_dim*(keypoint_prop)
            dst_pixel_limit = img_dim*(1 - keypoint_prop)
        else:
            src_pixel_limit = img_dim*(1 - keypoint_prop)
            dst_pixel_limit = img_dim*(keypoint_prop)
        
        if config["stitching_direction"] == 'LEFT':
            filtered_idx = np.where((m_kpts0_np[:,0] < src_pixel_limit) & (m_kpts1_np[:,0] > dst_pixel_limit))
        elif config["stitching_direction"] == 'UP':
            filtered_idx = np.where((m_kpts0_np[:,1] < src_pixel_limit) & (m_kpts1_np[:,1] > dst_pixel_limit))
        elif config["stitching_direction"] == 'RIGHT':
            filtered_idx = np.where((m_kpts0_np[:,0] > src_pixel_limit) & (m_kpts1_np[:,0] < dst_pixel_limit))
        else:
            filtered_idx = np.where((m_kpts0_np[:,1] < src_pixel_limit) & (m_kpts1_np[:,1] > dst_pixel_limit))
            
        if len(filtered_idx[0]) > 3:
            keypoint_prop_dict[keypoint_prop] = filtered_idx
    
    ####################################################
    #Filter based on keypoint distances from each other#
    ####################################################
    for keypoint_prop, filtered_idx in keypoint_prop_dict.items():
        #We want src points to be close to dst points in global space, so we exclude points that are
        #on the wrong part of the image to try to get the minimal images necessary and keypoints that are
        #unlikely to be present in more than two images.
        m_kpts0_f = m_kpts0_np[filtered_idx]
        m_kpts1_f = m_kpts1_np[filtered_idx]

        if len(m_kpts0_f) >= config["min_inliers"]:
            #Changing the RANSAC threshold parameter will determine if we get more but noisier matches (higher value)
            #or fewer but more pixel-perfect matches (lower value). Lower values help ensure that the OpenCV Matcher
            #will also match the points.
            transformation_matrix = None
            
            ######################################################
            #Filter based on RANSAC threshold and minimum inliers#
            ###################################################### 
            #Use lowest RANSAC threshold possible to meet minimum inliers
            for RANSAC_threshold in range(1, int(config["max_RANSAC_thresh"]) + 1):
                H, mask = cv2.findHomography(m_kpts0_f, m_kpts1_f, cv2.RANSAC, RANSAC_threshold)
                if np.sum(mask) >= config["min_inliers"]:
                    transformation_matrix = H
                    break
            ########################################
            #Filter based on homography constraints#
            ########################################
            if transformation_matrix is not None:
                if config["stitching_direction"] == 'LEFT':
                    stitch_movement = transformation_matrix[0, 2]
                    forward_vs_lateral = abs(transformation_matrix[0,2]/transformation_matrix[1,2])
                elif config["stitching_direction"] == 'UP':
                    stitch_movement = transformation_matrix[1, 2]
                    forward_vs_lateral = abs(transformation_matrix[1,2]/transformation_matrix[0,2])
                elif config["stitching_direction"] == 'RIGHT':
                    stitch_movement = -1*transformation_matrix[0, 2]
                    forward_vs_lateral = abs(transformation_matrix[0,2]/transformation_matrix[1,2])
                else:
                    stitch_movement = -1*transformation_matrix[1, 2]
                    forward_vs_lateral = abs(transformation_matrix[1,2]/transformation_matrix[0,2])
                scale = (transformation_matrix[0,0]**2 + transformation_matrix[1,0]**2)**0.5 #estimate scale factor
    
                #We only want matches where the homography matrix indicates that there is positive movement in the
                #stitching direction, there is more movement in the stitching direction that the normal, and that
                #the distance of the camera from the plane is not changing too much. We also need sufficient points
                #that match this homography or we risk the OpenCV Matcher failing to match the points.
                if (((stitch_movement > 0) and forward_vs_lateral > config["xy_ratio"]) and 
                    (abs(scale - 1.0) < config["scale_constraint"])):
                    k0_idx = matches[:,0].cpu().numpy()[filtered_idx][mask.astype(bool).flatten()]
                    k1_idx = matches[:,1].cpu().numpy()[filtered_idx][mask.astype(bool).flatten()]
                    preselect_kp0, preselect_feat0 = kpts0[k0_idx].cpu().numpy(), feats0['descriptors'][k0_idx].cpu().numpy()
                    preselect_kp1, preselect_feat1 = kpts1[k1_idx].cpu().numpy(), feats1['descriptors'][k1_idx].cpu().numpy()
                    mean_error, _ = get_LMEDS_error(preselect_kp0, preselect_kp1, config)
                    ####################################################################################################
                    #Filter out what we believe to be the most non-planar points so that we can optimize the homography#
                    ####################################################################################################
                    if config['refine']:
                        #Once we have excluded extreme outliers, try to find the best points to use to find the final homography matrix.
                        #Since RANSAC can quickly exclude points but may not find an optimal solution, even when the RANSAC threshold is low,
                        #we try to remove outliers using a brute force method.
                        """TO DO: This brute force method can probably be further optimized, but the cv2 implementation is fast enough
                        that it is not currently a problem"""
                        idx = np.arange(len(preselect_kp0)) #use to keep track of the final indices to keep
                        idx = idx[:, None] #make column vector
                        idx_kp0 = np.hstack((idx, preselect_kp0)) #add idx to the keypoints
                        idx_kp1 = np.hstack((idx, preselect_kp1))
                        maximum_removals = int(len(preselect_kp0) - config["min_inliers"])
                        error_array = full_outlier_function(preselect_kp0, preselect_kp1, maximum_removals, config)
                        if len(error_array) > 0:
                            best_iterations = np.argmin(error_array[:,1]) #Find best number of points to remove to minimize mean error
                            mean_error = error_array[best_iterations, 1]
                        else:
                            best_iterations = 0
                        #Recreate the outlier removal to recover the optimal set of points
                        idx_to_keep = incremental_outlier_removal(idx_kp0, idx_kp1, best_iterations, config)
                        preselect_kp0, preselect_feat0 = preselect_kp0[idx_to_keep], preselect_feat0[idx_to_keep]
                        preselect_kp1, preselect_feat1 = preselect_kp1[idx_to_keep], preselect_feat1[idx_to_keep]
                    ####################################
                    #Filter based on reprojection error#
                    ####################################
                    if mean_error <= config["max_reprojection_error"]:
                        return (img_feats, True, preselect_kp0, preselect_feat0, preselect_kp1, preselect_feat1, mean_error, RANSAC_threshold, keypoint_prop)
                #     else:
                #         print("Failed Error Filter", src_idx, dst_idx, mean_error, RANSAC_threshold, keypoint_prop)
                # else:
                #     print("Failed Homography Constraints", src_idx, dst_idx, RANSAC_threshold, keypoint_prop)
    ########################################################################
    #Force to match with image if this is there are no more images to check#
    ########################################################################
    if dst_idx - src_idx == 1:
        #If the filters exclude the match, pass the original match points in the most generous
        #RANSAC inliers and features as a last resort
        H, default_mask = cv2.findHomography(m_kpts0_np, m_kpts1_np, cv2.RANSAC, config["max_RANSAC_thresh"])
        k0_idx = matches[:,0].cpu().numpy()[default_mask.astype(bool).flatten()]
        k1_idx = matches[:,1].cpu().numpy()[default_mask.astype(bool).flatten()]
        
        preselect_kp0, preselect_feat0 = kpts0[k0_idx].cpu().numpy(), feats0['descriptors'][k0_idx].cpu().numpy()
        preselect_kp1, preselect_feat1 = kpts1[k1_idx].cpu().numpy(), feats1['descriptors'][k1_idx].cpu().numpy()
        mean_error, _ = get_LMEDS_error(preselect_kp0, preselect_kp1, config)
        return (img_feats, True, preselect_kp0, preselect_feat0, preselect_kp1, preselect_feat1, mean_error, None, 1.0)
    else:
        return (img_feats, False, None, None, None, None, None, None, 1.0)

def incremental_outlier_removal(pt0, pt1, iterations, config):
    ###################################################################
    #Find the points that should be removed to get the best mean error#
    ###################################################################
    #pt0 and pt1 should be [idx, x, y] so we can keep track of the 
    #final indices of the best points
    for i in range(iterations):
        #Pass the xy coordinates
        new_error, idx = get_LMEDS_error(pt0[:,1:], pt1[:,1:], config)
        idx_to_keep = idx[:-1]
        pt0, pt1 = pt0[idx_to_keep], pt1[idx_to_keep]
    return pt0[:,0].astype(np.int32) #Return the indices
        
def full_outlier_function(pt0, pt1, maximum_removals, config):
    ################################################################
    #Compute the mean reprojection error when we iteratively remove#
    #the biggest outlier from the original set of keypoints        #
    ################################################################
    #We assume that the images are non-planar and that the most planar points will show the lowest mean error, so we
    #try to exclude outlier points based on median error with the expectation that those points will tend to be the most
    #non planar points.
    #There should be a balance between having a high number of points to keep mean error low and removing
    #points that shift inconsistently with the other points so that the total error is low 
    error_array = [] #keep track of the outliers removed and mean error
    for i in range(maximum_removals):
        new_error, idx = get_LMEDS_error(pt0, pt1, config)
        if new_error == None:
            #If the homography cannot be found, stop the search
            return np.array(error_array)
        error_array.append([i, new_error])
        idx_to_keep = idx[:-1] #Drop the point with the highest error
        pt0, pt1 = pt0[idx_to_keep], pt1[idx_to_keep] #Re-run with the new points
    return np.array(error_array)

def get_LMEDS_error(kp0, kp1, config):
    ######################################################
    #Return the points ranked by their reprojection error#
    ######################################################
    #Calculate the optimal homography matrix using Least Median Robust Method
    #We expect that the median error will be minimized when excluding the most
    #non-planar point, which should have the highest reprojection error
    H_LMEDS, _LMEDS = cv2.findHomography(kp0, kp1, cv2.LMEDS)
        
    if H_LMEDS is None:
        #We assume that if no homography matrix can be found, there are already
        #too few points and we can stop searching
        return None, np.arange(len(kp0))
    else:
        #Calculate reprojection error as pixel distance between projected and dst
        rotations = np.matmul(kp0, H_LMEDS[:2, :2])
        rotations[:, 0] += H_LMEDS[0, 2]
        rotations[:, 1] += H_LMEDS[1, 2]
        difference = kp1 - rotations
        error = np.linalg.norm(difference, axis = 1)
        idx = np.argsort(error) #Return the sorted IDs of keypoints from best to worst
        return np.mean(error), idx

def check_forward_matches(img_matches, img_feats, img_paths, src_idx, img_dims, config):
    ######################################################################
    #Try to find matches with keypoints clustered on their stitching edge#
    ######################################################################
    #Need to know which part of the image we should expect src points to be on vs. dst points
    if config["stitching_direction"] == 'LEFT' or config["stitching_direction"] == 'RIGHT':
        img_dim = img_dims[0]
    else:
        img_dim = img_dims[1]
        
    ##############################################
    #Check  images starting with far images first#
    ###############################################
    for dst_idx in range(src_idx + config["forward_limit"], src_idx, -1):
        if len(img_paths) > dst_idx:
            ###################################
            #Match images based on constraints#
            ###################################
            """for this image, find lowest kp prop that meets criteria or skip"""
            img_feats, matched, kps, fs, kpd, fd, error, ransac, keypoint_prop = get_inliers(img_feats, img_paths,
                                                               src_idx, dst_idx,
                                                               img_dim, config)
            
            ###############################################
            #Save keypoints and features for use in OpenCV#
            ###############################################
            if matched:
                img_matches[src_idx]['keypoints']['src'] = [keypoint for keypoint in kps.tolist()]
                img_matches[src_idx]['features']['src'] = [feature for feature in fs.tolist()]
                img_matches[dst_idx]['keypoints']['dst'] = [keypoint for keypoint in kpd.tolist()]
                img_matches[dst_idx]['features']['dst'] = [feature for feature in fd.tolist()]
                print('Matched ', src_idx, dst_idx, ' with proportion limit: ', keypoint_prop,
                      len(kps), ' inliers with mean error: ', error, ' and RANSAC threshold: ', ransac)
                return img_matches, img_feats, dst_idx
        
    raise ValueError("Could not find a match with ", src_idx, " try lowering min_inliers, extracting more frames, or increasing forward_limit")

def find_matching_images(img_paths, config):
    ##########################################################
    #Using features extracted from SuperPoint, use LightGlue #
    #to find the minimum set of images that can be stitched  #
    # at high confidence and connect the first and last image#
    ##########################################################
    image = cv2.imread(img_paths[0]) #load first image to get dimensions
    dummy_img = cv2.resize(image, dsize = None, fx = config["feature_resolution"], fy = config["feature_resolution"])
    img_xdim, img_ydim = dummy_img.shape[1], dummy_img.shape[0]
    img_dims = (img_xdim, img_ydim)

    #A dictionary to hold matching src and dst keypoints and features between images
    img_matches = {}
    for i in range(len(img_paths)):
        img_matches[i] = {'keypoints': {'src': [], 'dst': []}, 'features': {'src': [], 'dst': []}}
        
    src_idx, dst_idx = 0, 0
    image_subset = [src_idx]
    #A dictionary to hold pointers to feature tensors extracted on GPU (or RAM), we avoid extracting the features from all images
    #because we ideally use as few images as possible
    img_feats = {}
    print('Finding best image matches...')
    while src_idx < len(img_paths) - 1:
        #only add keypoints and features to the dictionary if they are the best match for that image
        img_matches, img_feats, dst_idx = check_forward_matches(img_matches, img_feats, img_paths, src_idx, img_dims, config)
        src_idx = dst_idx
        image_subset.append(dst_idx)
        
    print('Using ', len(image_subset), ' images of the initial ', len(img_paths))
    print('Image subset: ', image_subset)
    return img_matches, image_subset

def build_feature_objects(subset_image_paths, img_matches, subset_indices, config):
    ##############################################################################
    #Convert the preselected SuperPoint keypoint and features into OpenCV objects#
    ##############################################################################
    cv_features = []
    image = cv2.imread(subset_image_paths[0]) #load first image to get dimensions
    dummy_img = cv2.resize(image, dsize = None, fx = config["feature_resolution"], fy = config["feature_resolution"])
    
    #We pass dummy images since we are manually setting the info
    for idx in subset_indices:
        feat = cv2.detail.computeImageFeatures2(cv2.ORB.create(), dummy_img)
        keypoints = np.array(img_matches[idx]['keypoints']['src'] + img_matches[idx]['keypoints']['dst'])
        feat.keypoints = tuple(cv2.KeyPoint(keypoints[x, 0], keypoints[x, 1], 0.0) for x in range(len(keypoints)))
        feat.descriptors = cv2.UMat(np.array(img_matches[idx]['features']['src'] + img_matches[idx]['features']['dst'], dtype = np.float32))
        cv_features.append(feat)
    
    return cv_features

def subset_images(image_paths, config):
    ##################################
    #Find best matches between images#
    ##################################
    img_matches, subset_indices = find_matching_images(image_paths, config)

    ###########################################################################
    #Use the matched images and keypoints to create the OpenCV feature objects#
    ###########################################################################
    subset_image_paths = [image_paths[i] for i in subset_indices]
    cv_features = build_feature_objects(subset_image_paths, img_matches, subset_indices, config)
    return cv_features, subset_indices, img_matches

def prepare_OpenCV_objects(config):
    #####################
    #Read in image paths#
    #####################
    path = config["image_directory"]
    image_paths = [os.path.join(path, img_name) for img_name in os.listdir(path)]
    print('Found ', len(image_paths), ' images')

    ###############################################################################
    #Get the features for the best subset of images using SuperPoint and LightGlue#
    ###############################################################################
    cv_features, subset_indices, img_matches = subset_images(image_paths, config)

    ############################
    #Calculate pairwise matches#
    ############################
    pairs = itertools.product(range(len(cv_features)), range(len(cv_features)))
    matches = []
    #Since we only want subsequent images matched, we only calculate matches between images and
    #the previous or next image, the other pairwise matches are set to have no matches
    #to prevent mismatches across images.
    for i, j in pairs:
        if abs(j - i) > 1 or i == j:
            match = cv2.detail.MatchesInfo()
            if i == j:
                #This matches convention of cv matching of self to self
                match.src_img_idx, match.dst_img_idx = -1, -1
            else:
                match.src_img_idx, match.dst_img_idx = i, j
            match.H = None
            match.confidence = 0.0
            match.inliers_mask = ()
            match.num_inliers = 0
            match.matches = ()
        else:
            #One issue is that the OpenCV matcher is not as good as the LightGlue one, so even though
            #we handpick keypoints and features that we know are good matches, it will not always
            #recognize that. This matcher uses Lowe's ratio test, where a low match_conf will allow more
            #points (I think this is a poor implementation or the documentation is wrong?).
            #Default is 0.3. matches_confidece_thresh (sic) is the threshold over which a match is 
            #considered a match, default is 3.0. If confident in the matches, set match_conf low to avoid
            #excluding true matches.
            matcher = cv2.detail.BestOf2NearestMatcher(try_use_gpu = False, match_conf = 0.1,
                                                         num_matches_thresh1 = 6, num_matches_thresh2 = 6,
                                                           matches_confindece_thresh = 3.0)


            #apply2 finds all pairwise matches and is accelerated by TBB, but we can beat that performance
            #serially by simply skipping most pairs
            match = matcher.apply(cv_features[i], cv_features[j])
            match.src_img_idx, match.dst_img_idx = i, j
        matches.append(match)

    #If there is poor confidence across images the stitching may fail, consider changing
    #the confidence thresholds or the minimum number of inliers
    full_confidence_matrix = np.array([match.confidence for match in matches]).reshape(len(cv_features), len(cv_features))
    sequential_matches = np.array([full_confidence_matrix[i, i+1] for i in range(len(cv_features)-1)])
    print('Confidence values of subsequent images: ')
    print(sequential_matches)
    if len(np.where(sequential_matches == 0)[0]) > 0:
        raise ValueError("Could not connect all images!")
        
    #######################################################################
    #Make list of the subset of images used and resize to final resolution#
    #######################################################################
    images = [cv2.resize(cv2.imread(image_paths[i]), dsize = None, fx = config["final_resolution"], fy = config["final_resolution"])
                                   for i in subset_indices]

    ########################################################################################
    #Make a dictionary with the src and dst keypoints and features for the subset of images#
    ########################################################################################
    subset_img_keypoints = {i: img_matches[k]['keypoints'] for i, k in enumerate(subset_indices)}
    return images, cv_features, matches, subset_img_keypoints

    

def spherical_OpenCV_pipeline(images, features, matches, config):
    #####################################################################
    #Process images assuming that the camera is stationary and rotating.#
    #####################################################################
    cameras = spherical_camera_estimation(features, matches, config)
    processed_images = spherical_warp_images(images, cameras, config)
    processed_images = get_seams_and_compensate(*processed_images)
    panorama = blend_images(*processed_images)
    return panorama

def spherical_camera_estimation(features, matches, config):
    ###########################################################################
    #Estimate camera rotations and focal length (can change across cameras),  #
    #with principalx and principaly constant across cameras and no translation#
    ###########################################################################
    print('Estimating cameras...')
    estimator = cv2.detail_HomographyBasedEstimator()
    success, cameras = estimator.apply(features, matches, None)
    if not success:
        raise ValueError("Failed to estimate cameras")
        
    print('Adjusting cameras...')
    #change types to match what bundleAdjuster wants
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
        
    adjuster = cv2.detail_BundleAdjusterRay()
    adjuster.setConfThresh(0.5)
    success, cameras =adjuster.apply(features, matches, cameras)
    if not success:
        raise ValueError("Failed to adjust cameras")
        
    return cameras

def spherical_warper(original_img, camera, scale, aspect_ratio):
    warper = cv2.PyRotationWarper("spherical", scale*aspect_ratio)
    w, h = original_img.shape[1], original_img.shape[0]
    K = camera.K().astype(np.float32)
    K[0, 0] *= aspect_ratio
    K[0, 2] *= aspect_ratio
    K[1, 1] *= aspect_ratio
    K[1, 2] *= aspect_ratio
    roi  = warper.warpRoi((w, h), K = K, R = camera.R) #returns (top_leftx, top_lefty, sizex, sizey)
    top_left, warped = warper.warp(original_img, K = K, R = camera.R,
                      interp_mode = cv2.INTER_LINEAR, border_mode = cv2.BORDER_REFLECT)
    #Create a black and white mask of the warped image for cropping, finding seams, and blending
    mask = 255 * np.ones((h, w), np.uint8)
    top_left, mask = warper.warp(mask, K = K, R = camera.R,
                      interp_mode = cv2.INTER_NEAREST, border_mode = cv2.BORDER_CONSTANT)
    return warped, mask, roi[0:2], roi[2:4]


def spherical_warp_images(images, cameras, config):
    print('Warping images with my spherical projection...')
    scale = np.median([cam.focal for cam in cameras])
    
    warped_final_imgs = []
    warped_final_masks = []
    final_corners = []
    final_sizes = []
    camera_aspect = config["final_resolution"]/config["feature_resolution"]

    for img, camera in zip(images, cameras):
        warped_img, warped_mask, corner, size = spherical_warper(img, camera, scale, camera_aspect)
        warped_final_imgs.append(warped_img)
        warped_final_masks.append(warped_mask)
        final_corners.append(corner)
        final_sizes.append(size)
        
    warped_low_imgs = []
    warped_low_masks = []
    low_corners = []
    low_sizes = []
    low_imgs = [cv2.resize(img, dsize = None, fx = config["seam_resolution"], fy = config["seam_resolution"]) for img in images]
    
    downscale_aspect_ratio = config["seam_resolution"]
    for img, camera in zip(low_imgs, cameras):
        warped_img, warped_mask, corner, size = spherical_warper(img, camera, scale, downscale_aspect_ratio)
        warped_low_imgs.append(warped_img)
        warped_low_masks.append(warped_mask)
        low_corners.append(corner)
        low_sizes.append(size)

    return (warped_low_imgs, warped_low_masks, low_corners, low_sizes,
            warped_final_imgs, warped_final_masks, final_corners, final_sizes)


def get_seams_and_compensate(low_imgs, low_masks, low_corners, low_sizes,
                             final_imgs, final_masks, final_corners, final_sizes):
    #######################################################################################
    #Find seams in overlapping areas and compensate the images for differences in exposure#
    #######################################################################################
    
    print('Finding seams...')    
    #colorgrad outperforms default color option, for higher resolution images seams are more aggressive
    #and more image is lost, colorgrad helps eliminate duplication without losing too much image
    seam_finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
    imgs = [img.astype(np.float32) for img in low_imgs]
    seam_masks = seam_finder.find(imgs, low_corners, low_masks)
    resized_seam_masks = [cv2.resize(seam_mask, (final_mask.shape[1], final_mask.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT) for seam_mask, final_mask in zip(seam_masks, final_masks)]
    final_seam_masks = [cv2.bitwise_and(resized_seam_mask, final_mask) for resized_seam_mask, final_mask in zip(resized_seam_masks, final_masks)]

    return final_seam_masks, final_imgs, final_corners, final_sizes

def blend_images(seam_masks, imgs, final_corners, final_sizes, blend_strength = 5):
    ###################################
    #Blend images together using seams#
    ###################################
    print('Preparing to blend images...')        
    # blender = Blender()
    # blender.prepare(final_corners, final_sizes)
    # for img, mask, corner in zip(imgs, seam_masks, final_corners):
    #     blender.feed(img, mask, corner)
    # print('Blending images...')
    # panorama, _ = blender.blend()
    # print('Finished stitching')
    
    dst_sz = cv2.detail.resultRoi(corners = final_corners, sizes = final_sizes)
    blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(int((np.log(blend_width) / np.log(2.0) - 1.0)))
    blender.prepare(dst_sz)
    
    for img, mask, corner in zip(imgs, seam_masks, final_corners):
        blender.feed(cv2.UMat(img.astype(np.int16)), mask, corner)
    
    blended, mask = blender.blend(None, None)
    panorama = cv2.convertScaleAbs(blended)
    
    return panorama
    
def load_config(config_path):
    ################################################################
    #Load configuration from a YAML file and compile regex patterns#
    ################################################################
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Adjust device setting based on the string from config
    config["device"] = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    print("Configuration loaded successfully...")
    return config
    
def run_stitching_pipeline(config_path):
    config = load_config(config_path)
    images, cv_features, matches, keypoint_dict = prepare_OpenCV_objects(config)
    try:
        panorama = spherical_OpenCV_pipeline(images, cv_features, matches, config)
    except (ValueError):
        print("Spherical projection failed")

    if config['save_image']:
        print('Saving image...')
        cv2.imwrite(os.path.join(config['output_dir'], config['output_filename']), panorama)
    
if __name__ == "__main__":
    config_path = sys.argv[1]
    run_stitching_pipeline(config_path)