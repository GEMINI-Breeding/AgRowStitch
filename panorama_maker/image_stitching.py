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
import scipy as sp
import sys
import torch
from torchvision import transforms
import yaml


###############################################################################
#                              SINGLE PANORAMA                                #
###############################################################################


def extract_features(img_path, config):
    ##################################
    #Load image and reduce resolution#
    ##################################
    image = cv2.imread(img_path) #Read image and then resize
    medium_img = cv2.resize(image, dsize = None, fx = config["final_resolution"], fy = config["final_resolution"])
    if config["stitching_direction"] == 'RIGHT':
        #Flip image across vertical axis so the stitching edge is now on the left
        medium_img = cv2.flip(medium_img, 1)
    elif config["stitching_direction"] == 'UP':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        medium_img = cv2.rotate(medium_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif config["stitching_direction"] == 'DOWN':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        medium_img = cv2.rotate(medium_img, cv2.ROTATE_90_CLOCKWISE)

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
        src_pixel_limit = img_dim*(keypoint_prop)
        dst_pixel_limit = img_dim*(1 - keypoint_prop)
        filtered_idx = np.where((m_kpts0_np[:,0] < src_pixel_limit) & (m_kpts1_np[:,0] > dst_pixel_limit))
        #Need at least four points to find a transform
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
                stitch_movement = transformation_matrix[0, 2]
                forward_vs_lateral = abs(transformation_matrix[0,2]/transformation_matrix[1,2])
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
    img_dim = img_dims[0]
    ##############################################
    #Check  images starting with far images first#
    ###############################################
    for dst_idx in range(src_idx + config["forward_limit"], src_idx, -1):
        if len(img_paths) > dst_idx:
            ###################################
            #Match images based on constraints#
            ###################################
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
                return img_matches, img_feats, dst_idx, ransac
    raise ValueError("Could not find a match with ", src_idx, " try lowering min_inliers, extracting more frames, or increasing forward_limit")

def find_matching_images(img_paths, start_idx, config):
    ##########################################################
    #Using features extracted from SuperPoint, use LightGlue #
    #to find the minimum set of images that can be stitched  #
    # at high confidence and connect the first and last image#
    ##########################################################
    image = cv2.imread(img_paths[0]) #load first image to get dimensions
    dummy_img = cv2.resize(image, dsize = None, fx = config["final_resolution"], fy = config["final_resolution"])
    if config["stitching_direction"] == 'RIGHT':
        #Flip image across vertical axis so the stitching edge is now on the left
        dummy_img = cv2.flip(dummy_img, 1)
    elif config["stitching_direction"] == 'UP':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        dummy_img = cv2.rotate(dummy_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif config["stitching_direction"] == 'DOWN':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        dummy_img = cv2.rotate(dummy_img, cv2.ROTATE_90_CLOCKWISE)
        
    img_xdim, img_ydim = dummy_img.shape[1], dummy_img.shape[0]
    img_dims = (img_xdim, img_ydim)
    #A dictionary to hold matching src and dst keypoints and features between images
    img_matches = {}
    for i in range(start_idx, len(img_paths)):
        img_matches[i] = {'keypoints': {'src': [], 'dst': []}, 'features': {'src': [], 'dst': []}}
    src_idx, dst_idx = start_idx, start_idx
    image_subset = [start_idx]
    #A dictionary to hold pointers to feature tensors extracted on GPU (or RAM), we avoid extracting the features from all images
    #because we ideally use as few images as possible
    img_feats = {}
    filtered = True #We keep track of whether the matcher had to default to using the raw keypoints for at least one of the images
    while src_idx < len(img_paths) - 1 and len(image_subset) < config["batch_size"]:
        #Only add keypoints and features to the dictionary if they are the best match for that image
        img_matches, img_feats, dst_idx, ransac = check_forward_matches(img_matches, img_feats, img_paths, src_idx, img_dims, config)
        if ransac is None:
            filtered = False
        src_idx = dst_idx
        image_subset.append(dst_idx)
    print('Using ', len(image_subset), ' images of the initial ', image_subset[-1] - image_subset[0] + 1)
    return img_matches, image_subset, filtered

def build_feature_objects(subset_image_paths, img_matches, subset_indices, config):
    ##############################################################################
    #Convert the preselected SuperPoint keypoint and features into OpenCV objects#
    ##############################################################################
    cv_features = []
    image = cv2.imread(subset_image_paths[0]) #load first image to get dimensions
    dummy_img = cv2.resize(image, dsize = None, fx = config["final_resolution"], fy = config["final_resolution"])
    if config["stitching_direction"] == 'RIGHT':
        #Flip image across vertical axis so the stitching edge is now on the left
        dummy_img = cv2.flip(dummy_img, 1)
    elif config["stitching_direction"] == 'UP':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        dummy_img = cv2.rotate(dummy_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif config["stitching_direction"] == 'DOWN':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        dummy_img = cv2.rotate(dummy_img, cv2.ROTATE_90_CLOCKWISE)
    #We pass dummy images since we are manually setting the info
    for idx in subset_indices:
        feat = cv2.detail.computeImageFeatures2(cv2.ORB.create(), dummy_img)
        keypoints = np.array(img_matches[idx]['keypoints']['src'] + img_matches[idx]['keypoints']['dst'])
        feat.keypoints = tuple(cv2.KeyPoint(keypoints[x, 0], keypoints[x, 1], 0.0) for x in range(len(keypoints)))
        feat.descriptors = cv2.UMat(np.array(img_matches[idx]['features']['src'] + img_matches[idx]['features']['dst'], dtype = np.float32))
        cv_features.append(feat)
    return cv_features

def subset_images(image_paths, start_idx, config):
    ##################################
    #Find best matches between images#
    ##################################
    img_matches, subset_indices, filtered = find_matching_images(image_paths, start_idx, config)
    ###########################################################################
    #Use the matched images and keypoints to create the OpenCV feature objects#
    ###########################################################################
    subset_image_paths = [image_paths[i] for i in subset_indices]
    cv_features = build_feature_objects(subset_image_paths, img_matches, subset_indices, config)
    return cv_features, subset_indices, img_matches, filtered

def prepare_OpenCV_objects(start_idx, config):
    #####################
    #Read in image paths#
    #####################
    path = config["image_directory"]
    image_paths = [os.path.join(path, img_name) for img_name in os.listdir(path)]
    if start_idx == 0:
        print('Found ', len(image_paths), ' images')
    ###############################################################################
    #Get the features for the best subset of images using SuperPoint and LightGlue#
    ###############################################################################
    cv_features, subset_indices, img_matches, filtered = subset_images(image_paths, start_idx, config)
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
    # print('Confidence values of subsequent images: ')
    # print(sequential_matches)
    if len(np.where(sequential_matches == 0)[0]) > 0:
        raise ValueError("Could not connect all images!")
    #######################################################################
    #Make list of the subset of images used and resize to final resolution#
    #######################################################################
    images = [cv2.resize(cv2.imread(image_paths[i]), dsize = None, fx = config["final_resolution"], fy = config["final_resolution"])
                                   for i in subset_indices]
    if config["stitching_direction"] == 'RIGHT':
        #Flip image across vertical axis so the stitching edge is now on the left
        images = [cv2.flip(image, 1) for image in images]
    elif config["stitching_direction"] == 'UP':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        images = [cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE) for image in images]
    elif config["stitching_direction"] == 'DOWN':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        images = [cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) for image in images]
    
    ########################################################################################
    #Make a dictionary with the src and dst keypoints and features for the subset of images#
    ########################################################################################
    subset_img_keypoints = {i: img_matches[k]['keypoints'] for i, k in enumerate(subset_indices)}
    if subset_indices[-1] >= len(image_paths) - 1:
        finished = True
    else:
        finished = False
    return images, cv_features, matches, subset_img_keypoints, subset_indices[-1],filtered, finished

def spherical_OpenCV_pipeline(images, features, matches, config):
    #############################################################################################
    #Process images assuming that the camera is stationary and rotating.                        #
    #Then project the images onto a sphere. The bundle adjustment process becomes               #
    #too computationally intensive (and unconstrained) when applied to a large number of images.#
    #However, the rotational DOF allow for high quality mostly planar projections for small     #
    #batches of images even when the camera is translating.                                     #
    #############################################################################################
    cameras = spherical_camera_estimation(features, matches, config)
    processed_images = spherical_warp_images(images, cameras, config)
    processed_images = get_seams(*processed_images)
    panorama = blend_images(*processed_images)
    return panorama

def bundle_affine_OpenCV_pipeline(images, features, matches, config):
    ############################################################################################
    #Process images assuming that the camera can translate and rotate.                         #
    #The OpenCV bundle adjustment procedure can lead to worse results                          #
    #because it uses the features and matched keypoints to try to find the                     #
    #camera positions that minimize error, but are thus prone to undoing the filtering process #
    #Since the behavior is unpredictable, this is not a recommended.                           #
    ############################################################################################
    cameras = affine_camera_adjustment(features, matches, config)
    processed_images = affine_warp_images(images, cameras, config)
    processed_images = get_seams(*processed_images)
    panorama = blend_images(*processed_images)
    return panorama

def affine_OpenCV_pipeline(images, keypoint_dict, translation_only, config):
    #########################################################################
    #Process images assuming that the camera can translate and rotate.      #
    #We calculate the affine transform directly from the keypoints and then #
    #use OpenCV for the seams and blending. Since there is no bundle        #
    #adjustment, the run time should be manageable and the results should be# 
    #stable as long as the matches are good.                                #
    #########################################################################
    if translation_only:
        cameras = estimate_translation_cameras(keypoint_dict, config)
    else:
        cameras = estimate_cameras(keypoint_dict, config)
    processed_images = affine_warp_images(images, cameras, config)
    processed_images = get_seams(*processed_images)
    panorama = blend_images(*processed_images)
    return panorama

def spherical_camera_estimation(features, matches, config):
    ###########################################################################
    #Estimate camera rotations and focal length (can change across cameras),  #
    #with principalx and principaly constant across cameras and no translation#
    ###########################################################################
    estimator = cv2.detail_HomographyBasedEstimator()
    success, cameras = estimator.apply(features, matches, None)
    if not success:
        raise ValueError("Failed to estimate cameras")
    #Change types to match what bundleAdjuster wants
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)  
    adjuster = cv2.detail_BundleAdjusterRay()
    #Having a low threshold helps force the cameras to keep the matches we want,
    #we assume that this is preferable to OpenCV trying to reject some of our image matches
    adjuster.setConfThresh(0.1)
    success, cameras =adjuster.apply(features, matches, cameras)
    if not success:
        raise ValueError("Failed to adjust cameras")
    #To help maintain straighter panoramas, use wave correction to help account for
    #the camera angle changing and not being normal to the ground
    wave_direction = cv2.detail.WAVE_CORRECT_HORIZ
    rotation_matrices = [np.copy(cam.R) for cam in cameras]
    rotation_matrices = cv2.detail.waveCorrect(rotation_matrices, wave_direction)
    for i, cam in enumerate(cameras):
        cam.R = rotation_matrices[i]
    return cameras

def spherical_warper(original_img, camera, scale, aspect_ratio):
    ##########################################################################
    #Project images onto a sphere assuming a stationary camera with rotations#
    #and variable focal length                                               #
    ##########################################################################
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
    ###################################################################
    #Warp images, find the position of their top left corners         #
    #in the final global coordinates, and create masks for the images.#
    ###################################################################
    #Focal distance has to be accounted for as this is a scale parameter that
    #interacts with the camera aspect (which will change based on resolution)
    scale = np.median([cam.focal for cam in cameras])
    warped_final_imgs = []
    warped_final_masks = []
    final_corners = []
    final_sizes = []
    camera_aspect = 1.0
    for img, camera in zip(images, cameras):
        warped_img, warped_mask, corner, size = spherical_warper(img, camera, scale, camera_aspect)
        warped_final_imgs.append(warped_img)
        warped_final_masks.append(warped_mask)
        final_corners.append(corner)
        final_sizes.append(size)
    #We create low resolution versions for seam finding
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

def affine_camera_adjustment(features, matches, config):
    ####################################################################
    #Estimate and adjust affine matrices in global coordinate          # 
    #using bundle adjustment. Since bundle adjustment is a complicated #
    #global minimization problem, this becomes unstable when there are #
    #many images. The process uses the features and matches to optimize#
    #transformations, but since it ignores the constraints we enforced #
    #upstream, it can exhibit poor behavior.                           #
    ####################################################################
    #Estimates affine transforms from matches and features
    estimator = cv2.detail_AffineBasedEstimator()
    success, cameras = estimator.apply(features, matches, None)
    if not success:
        raise ValueError("Failed to estimate cameras")
    #change types to match what bundleAdjuster wants
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
    #Changing confidence threshold may help adjustment if the optimization is difficult 
    #and fails. It should be easier if there is high confidence between subsequent images
    #and the correct adjustment (affine or spherical) is chosen. Lower the confidence threshold
    #to pass adjustment, but adjustment might be more error prone.
    adjuster = cv2.detail_BundleAdjusterAffinePartial()
    # adjuster = cv2.detail_BundleAdjusterAffinePartial() #removes shearing
    adjuster.setConfThresh(0.1)
    success, cameras =adjuster.apply(features, matches, cameras)
    if not success:
        raise ValueError("Failed to adjust cameras")
    return cameras

def estimate_cameras(keypoint_dict, config):
    ##########################################################
    #Use the keypoints to recalculate transformation matrices#
    #according to the camera option chosen in congfig.       #
    ##########################################################
    #First estimate the partial affine matrix between subsequent images
    #using LMEDS. Since the points have already been filtered with RANSAC
    #and the LMEDS outlier removal, we assume that the LMEDS estimate
    #will be stable.
    num_images = len(keypoint_dict)
    H_pairs = [] #homography matrix from src to dst for subsequent images
    for src, dst in [(i, i+1) for i in range(num_images - 1)]:
        kp0 = np.array(keypoint_dict[src]['src'], dtype = np.int64)
        kp1 = np.array(keypoint_dict[dst]['dst'], dtype = np.int64)
        if config["camera"] == "homography":
            #Full homography is less stable than affine and partial affine
            #because perspective changes on non-planar objects
            #will distort objects based on depth and may lead to 
            #cliping images during warping. Poor homography matches
            #can lead to memory issues and failure.
            H_LMEDS, _LMEDS = cv2.findHomography(kp0, kp1, cv2.LMEDS)
        elif config["camera"] == "affine":
            #Affine transformations are more stable than homography, but 
            #the shearing can also distort the final panorama.
            H_LMEDS, _LMEDS = cv2.estimateAffine2D(kp0, kp1, cv2.LMEDS)
            H_LMEDS = np.vstack((H_LMEDS, np.array([0, 0, 1])))
        else: 
            #Partial affine transformations are the most stable and are recommended
            #unless the objects are planar (homography) or high shearing (affine).
            H_LMEDS, _LMEDS = cv2.estimateAffinePartial2D(kp0, kp1, cv2.LMEDS)
            H_LMEDS = np.vstack((H_LMEDS, np.array([0, 0, 1])))
        H_pairs.append(H_LMEDS)
    #Now choose the middle image as a reference image and convert the transformation
    #matrices to be with resepct to the reference to create global coordinates.
    H_list = []
    middle_idx = num_images//2 #Use middle image as the reference image
    for i in range(num_images):
        H_to_middle = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])
        #Build matrix to get to middle image for ith image
        if i < middle_idx:
            for m in range(middle_idx - 1, i - 1, -1):
            #Need to get inverse since the H is dst->src and going away from middle image
                H_to_middle = np.matmul(H_to_middle, np.linalg.inv(H_pairs[m]))
            H_list.append(H_to_middle)
        elif i == middle_idx:
            H_list.append(H_to_middle)
        else:
            for m in range(middle_idx, i, 1):
            #Keep H as is since dst ->src gets us towards middle image
                H_to_middle = np.matmul(H_to_middle, H_pairs[m])
            H_list.append(H_to_middle)
    #Save the matrix as a camera parameter to make it more accessible
    #to OpenCV functions in the future
    cameras = []
    for c in range(num_images):
        cam = cv2.detail.CameraParams()
        cam.R = H_list[c].astype(np.float32)
        cameras.append(cam)
    return cameras

def estimate_translation_cameras(keypoint_dict, config):
    ##########################################################
    #Use the keypoints to recalculate transformation matrices#
    #according to the camera option chosen in congfig.       #
    ##########################################################
    #First estimate the partial affine matrix between subsequent images
    #using LMEDS. Since the points have already been filtered with RANSAC
    #and the LMEDS outlier removal, we assume that the LMEDS estimate
    #will be stable.
    num_images = len(keypoint_dict)
    H_pairs = [] #homography matrix from src to dst for subsequent images
    for src, dst in [(i, i+1) for i in range(num_images - 1)]:
        kp0 = np.array(keypoint_dict[src]['src'], dtype = np.int64)
        kp1 = np.array(keypoint_dict[dst]['dst'], dtype = np.int64)
        H_LMEDS, _LMEDS = cv2.estimateAffinePartial2D(kp0, kp1, cv2.LMEDS)
        transx = H_LMEDS[0, 2]
        transy = H_LMEDS[1, 2]
        #We only keep the translations and assume that the scales should be
        #approximately the same because the median scale for each panorama
        #should be about the same
        reduced = np.array([[1, 0, transx],
                            [0, 1, transy]])
        H_LMEDS = np.vstack((reduced, np.array([0, 0, 1])))
        H_pairs.append(H_LMEDS)
    #Now choose the middle image as a reference image and convert the transformation
    #matrices to be with resepct to the reference to create global coordinates.
    H_list = []
    middle_idx = num_images//2 #Use middle image as the reference image
    for i in range(num_images):
        H_to_middle = np.array([[1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]])
        #Build matrix to get to middle image for ith image
        if i < middle_idx:
            for m in range(middle_idx - 1, i - 1, -1):
            #Need to get inverse since the H is dst->src and going away from middle image
                H_to_middle = np.matmul(H_to_middle, np.linalg.inv(H_pairs[m]))
            H_list.append(H_to_middle)
        elif i == middle_idx:
            H_list.append(H_to_middle)
        else:
            for m in range(middle_idx, i, 1):
            #Keep H as is since dst ->src gets us towards middle image
                H_to_middle = np.matmul(H_to_middle, H_pairs[m])
            H_list.append(H_to_middle)
    #Save the matrix as a camera parameter to make it more accessible
    #to OpenCV functions in the future
    cameras = []
    for c in range(num_images):
        cam = cv2.detail.CameraParams()
        cam.R = H_list[c].astype(np.float32)
        cameras.append(cam)
    return cameras

def affine_warp_images(images, cameras, config):
    ##################################################################################################
    #Get the transformed images and their masks as well as their start corners in global coordinates.#
    #Repeat for low resolution images so processing is easier downstream                             #
    ##################################################################################################
    #First work on the final resolution
    warped_final_imgs = []
    warped_final_masks = []
    final_corners = []
    final_sizes = []
    camera_aspect = 1.0
    for img, camera in zip(images, cameras):
        warped_img, warped_mask, corner, size = affine_warper(img, camera, camera_aspect)
        warped_final_imgs.append(warped_img)
        warped_final_masks.append(warped_mask)
        final_corners.append(corner)
        final_sizes.append(size)
    #We create low resolution versions for seam finding but scale the transformation matrix
    #directly to downscale the images rather than resize and then warp them
    warped_low_imgs = []
    warped_low_masks = []
    low_corners = []
    low_sizes = []
    downscale_aspect_ratio = config["seam_resolution"]
    for img, camera in zip(images, cameras):
        warped_img, warped_mask, corner, size = affine_warper(img, camera, downscale_aspect_ratio)
        warped_low_imgs.append(warped_img)
        warped_low_masks.append(warped_mask)
        low_corners.append(corner)
        low_sizes.append(size)
    return (warped_low_imgs, warped_low_masks, low_corners, low_sizes,
            warped_final_imgs, warped_final_masks, final_corners, final_sizes)

def warpROI(original_img, camera, aspect_ratio):
    ##################################################################
    #Find top left corner of warped image in global coordinates      #
    #and return the homography matrix adjusted into local coordinates#
    ##################################################################
    H = np.linalg.inv(camera.R) * aspect_ratio
    w, h = original_img.shape[1], original_img.shape[0]
    x, y = 0, 0
    #Corners (top left, bottom left, top right, bottom right) of original image in local coordinates
    corners = np.array([[x, y, 1], [x, y + h - 1, 1], [x + w - 1, y, 1], [x + w - 1, y + h - 1, 1]])
    top_left = np.floor(np.matmul(H, corners[0])[:2])
    bottom_left = np.floor(np.matmul(H, corners[1])[:2])
    top_right = np.floor(np.matmul(H, corners[2])[:2])
    bottom_right =  np.floor(np.matmul(H, corners[3])[:2])
    scaled_corners = np.array([top_left, bottom_left, top_right, bottom_right])
    minx, miny = np.min(scaled_corners[:, 0]), np.min(scaled_corners[:, 1])
    #Get the dimensions of the rectangular bounding box of the warped image, the top left corner of the bounding box is minx, miny
    global_top_left = (int(minx), int(miny)) 
    #Now remove the global translation from the homography matrix to make the homography in place
    T = np.array([[1, 0, -minx/H[2, 2]], [0, 1, -miny/H[2, 2]], [0, 0, 1]])
    H_global_adj = T.dot(H) #This has the global translation of the top left corner removed
    return global_top_left, H_global_adj

def affine_warper(original_img, camera, aspect_ratio):
    ###############################################################################
    #Warp image in local coordinates and the top left corner in global coordinates#
    ###############################################################################
    #Get top left corner of the warped image for global placement
    pos, H = warpROI(original_img, camera, aspect_ratio)
    w, h = original_img.shape[1], original_img.shape[0]
    x, y = 0, 0
    #Corners (top left, bottom left, top right, bottom right) of original image in local coordinates
    corners = np.array([[x, y, 1], [x, y + h - 1, 1], [x + w - 1, y, 1], [x + w - 1, y + h - 1, 1]])
    #Now we need to adjust the homography matrix again to translate the image to make sure 
    #all points are positive
    top_left = np.floor(np.matmul(H, corners[0])[:2])
    bottom_left = np.floor(np.matmul(H, corners[1])[:2])
    top_right = np.floor(np.matmul(H, corners[2])[:2])
    bottom_right =  np.floor(np.matmul(H, corners[3])[:2])
    scaled_corners = np.array([top_left, bottom_left, top_right, bottom_right])
    minx, miny = np.min(scaled_corners[:, 0]), np.min(scaled_corners[:, 1])
    maxx, maxy = np.max(scaled_corners[:, 0]), np.max(scaled_corners[:, 1])
    width = int(np.ceil(maxx - minx))
    height = int(np.ceil(maxy - miny))
    local_x_translation = int(-minx if minx < 0 else 0)
    local_y_translation = int(-miny if miny < 0 else 0)
    T = np.array([[1, 0, local_x_translation/H[2, 2]], [0, 1, local_y_translation/H[2, 2]], [0, 0, 1]])
    H_local_adj = T.dot(H)
    #This allows us to scale the image by aspect ratio with warpPerspective rather than generating the image at a lower resolution,
    #since everything else will be scaled by the aspect_ratio
    H_local_adj[2, 2] = 1 
    #Use warpPerspective with dst image to avoid potential memory leak when using dst = cv2.warpPerspective()
    warped = np.zeros((height, width, original_img.shape[2]), dtype = np.uint8)
    cv2.warpPerspective(original_img, H_local_adj, (width, height), warped, cv2.INTER_LINEAR)
    #Create a black and white mask for the warped image to help with stitching
    mask = 255 * np.ones((h, w), np.uint8)
    warped_mask = np.zeros((height, width), dtype = np.uint8)
    cv2.warpPerspective(mask, H_local_adj, (width, height), warped_mask, cv2.INTER_NEAREST)
    return warped, warped_mask, pos, (width, height)

def get_seams(low_imgs, low_masks, low_corners, low_sizes,
              final_imgs, final_masks, final_corners, final_sizes):
    #######################################################################################
    #Find seams in overlapping areas and compensate the images for differences in exposure#
    #######################################################################################
    #Colorgrad outperforms default color option, for higher resolution images seams are more aggressive
    #and more image is lost, colorgrad helps eliminate duplication without losing too much image
    seam_finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
    imgs = [img.astype(np.float32) for img in low_imgs]
    seam_masks = seam_finder.find(imgs, low_corners, low_masks)
    resized_seam_masks = [cv2.resize(seam_mask, (final_mask.shape[1], final_mask.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
                          for seam_mask, final_mask in zip(seam_masks, final_masks)]
    final_seam_masks = [cv2.bitwise_and(resized_seam_mask, final_mask)
                        for resized_seam_mask, final_mask in zip(resized_seam_masks, final_masks)]
    return final_seam_masks, final_imgs, final_corners, final_sizes

def blend_images(seam_masks, imgs, final_corners, final_sizes, blend_strength = 5):
    ###################################
    #Blend images together using seams#
    ###################################
    #Band number taken from open stitching 
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

def check_panorama(panorama, config):
    ####################################
    #Test for a major stitching failure#
    ####################################
    #Threshold the panorama to check its shape
    imgray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, 0) #separate black from non-black pixels
    ##################
    #Check dimensions#
    ##################
    #Since we do not have an expectation for the distance traveled by the camera, but we 
    #assume that the camera has minimal rotation, the dimension of the panorama in the non-stitching direction
    #should be relatively constant. If the maximum/median value is to high, we assume there was a poor
    #stitch
    dim = np.sum(thresh, axis = 0)
    dim_ratio = np.max(dim)/np.median(dim)
    ############################
    #Check if panorama has gaps#
    ############################
    #We assume panorama should consist of a single continuous image, so if there
    #are multiple contours (the image consists of multiple continuous images),
    #the panorama has failed.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ##############################################
    #Report whether the panorama passed the check#
    ##############################################
    if len(contours) == 1 and dim_ratio < 1.5:
        return True
    else:
        if len(contours) > 1.5:
            print("Panorama is not continuous")
        if dim_ratio > 1.5:
            print("Aspect ratio of panorama is concerning")
        return False

def retry_panorama(start_idx, filtered, config):
    ##############################################################
    #If a panorama cannot be created we try to adjust settings to# 
    #successfully generate the panorama                          #
    ##############################################################
    if not filtered:
        #################################################################
        #If the panorama was not filtered, it means that the constraints# 
        #imposed on the homography could not be met, so we try to relax #
        #those constraints to maintain some filter on the points        #
        #################################################################
        print("Retrying with relaxed error and inlier constraints...")
        config["max_reprojection_error"] *= 1.2
        config["min_inliers"] *= 0.8
        images, cv_features, matches, keypoint_dict, idx, filtered, finished = prepare_OpenCV_objects(start_idx, config)
        
        #If a spherical stitch is not possible, try with a partial affine stitch instead
        if config["camera"] == "spherical":
            try:
                new_panorama = spherical_OpenCV_pipeline(images, cv_features, matches, config)
                if not check_panorama(new_panorama, config):
                    print("Spherical stitching was unreliable, re-trying with partial affine...")
                    new_panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
            except:
                print("Spherical stitching failed, re-trying with partial affine...")
                new_panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
        else:
            new_panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
        #Return parameters to original values since the dictionary is modified in place
        config["max_reprojection_error"] /= 1.2
        config["min_inliers"] /= 0.8
    else:
        ##########################################################################
        #If the panorama was filtered, it means that the constraints were met,   #
        #so we tighten the constraints to try to remove outliers that could cause#
        #a stitching failure                                                     #
        ##########################################################################
        print("Panorama is unreliable, retrying with stronger error and inlier constraints...")
        config["max_reprojection_error"] *= 0.8
        config["min_inliers"] *= 1.2
        images, cv_features, matches, keypoint_dict, idx, filtered, finished = prepare_OpenCV_objects(start_idx, config)
        #If a spherical stitch is not possible, try with a partial affine stitch instead
        if config["camera"] == "spherical":
            try:
                new_panorama = spherical_OpenCV_pipeline(images, cv_features, matches, config)
                if not check_panorama(new_panorama, config):
                    print("Spherical stitching was unreliable, re-trying with partial affine...")
                    new_panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
            except:
                print("Spherical stitching failed, re-trying with partial affine...")
                new_panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
        else:
            new_panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
        #Return parameters to original values since the dictionary is modified in place
        config["max_reprojection_error"] /= 0.8
        config["min_inliers"] /= 1.2
    ###############################################
    #Check whether the new panorama was successful#
    ###############################################
    if check_panorama(new_panorama, config):
        return new_panorama, idx, finished
    else:
        return None, None, None

def run_stitching_pipeline(start_idx, config):
    ###############################################################################
    #Create panoramas by taking a subset of the images and stitching them together#
    #when the number of stitched images reaches the batch size                    #
    ###############################################################################
    #Extract the features and matches between the minimal subset of images, which are now ready to be stitched
    images, cv_features, matches, keypoint_dict, idx, filtered, finished = prepare_OpenCV_objects(start_idx, config)
    #########################################################################################
    #Spherical projection works best because of robust bundle adjustment and wave correction#
    #########################################################################################
    if config["camera"] == "spherical":
        try: #Since bundle adjustment can fail, we use a try/except statement
            panorama = spherical_OpenCV_pipeline(images, cv_features, matches, config)
            #If the panorama seems incorrect, use the same keypoints and try with a partial affine projection
            if not check_panorama(panorama, config):
                print("Spherical stitching was unreliable, re-trying with partial affine...")
                config["camera"] = "partial affine"
                panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
                config["camera"] = "spherical"
        except: #If bundle adjustment fails, fall back on a partial affine stitch with same keypoints instead
            print("Spherical stitching failed, re-trying with partial affine...")
            config["camera"] = "partial affine"
            panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
            config["camera"] = "spherical"
    ##########################################################################################
    #Partial affine recommended for mostly linear camera translation and minimal rotation,   #
    #can fail with many images and does not have bundle adjustment                           #
    #Homography and affine are not recommended because they can make unstable transformations#
    ##########################################################################################
    else:
        panorama = affine_OpenCV_pipeline(images, keypoint_dict, False, config)
    ##############################################################################################
    #Check whether the panorama needs to be attempted again with different keypoints and features#
    ##############################################################################################
    parent_dir = os.path.dirname(os.path.normpath(config["image_directory"]))
    output_dir = os.path.basename(os.path.normpath(config["image_directory"])) + '_output'
    output_path = os.path.join(parent_dir, output_dir)
    output_filename = 'batch_' + os.path.basename(os.path.normpath(config["image_directory"])) + '.png'
    if check_panorama(panorama, config):
        print(f'Saving image {idx} ...')
        cv2.imwrite(os.path.join(output_path, str(idx) + "_"  + output_filename), panorama)
        return finished, idx, True
    else:
        ##############################################################
        #If the current constraints are unable to produce a panorama,#
        #try with modified constraints                               #
        ##############################################################
        new_panorama, new_idx, new_finished = retry_panorama(start_idx, filtered, config)
        #Use new panorama if it passed the check
        if new_panorama is not None:
            print(f'Saving image {idx} ...')
            cv2.imwrite(os.path.join(output_path, str(idx) + "_"  + output_filename), new_panorama)
            return new_finished, new_idx, True
        #Otherwise, report a failure and continue
        else:
            return finished, idx, False
    
    
###############################################################################
#                              SUPER PANORAMA                                 #
###############################################################################


def extract_all_panorama_features(images, search_distance, config):
    ##################################################################
    #Use SuperPoint to extract keypoints and features from panoramas#
    #################################################################
    img_feats = {} #Dictionary to store features and keypoints
    for i in range(len(images)):
        img_feats[i] = {'src': [], 'dst': []}
    ##################################################################
    #Crop photo into a src and dst image that are pad pixels long    #
    #in the stitching direction so we can only extract keypoints from#
    #the relevant stitched edges of the panoramas                    #
    ##################################################################
    for i, image in enumerate(images):
        #Crop images using OpenCV index of height, width, but when we get the
        #keypoints back they will be in width, height format
        height, width = image.shape[0], image.shape[1]
        src_img = image[:, :search_distance, :]
        src_pad_array = np.array([0, 0])
        dst_img = image[:, (width - search_distance ):(width), :]
        dst_pad_array = np.array([width - search_distance , 0])

        ####################################
        #Use SuperPoint to extract features#
        ####################################
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(config["device"])
        src_image_tensor = transforms.ToTensor()(src_img).to(config["device"]).unsqueeze(0)
        dst_image_tensor = transforms.ToTensor()(dst_img).to(config["device"]).unsqueeze(0)
        with torch.no_grad():
            src_feats = extractor.extract(src_image_tensor)
            src_feats['scores'] = src_feats.get('scores', torch.ones((1, src_feats['keypoints'].shape[1]), device=src_feats['keypoints'].device))
            src_feats = rbd(src_feats)
            src_feats['keypoints'] = src_feats['keypoints'].unsqueeze(0)
            #Now we need to transform the keypoint coordinates to the full panorama coordinates
            src_pad_tensor = torch.from_numpy(np.tile(src_pad_array, (src_feats['keypoints'].shape[1], 1))).to(config["device"])
            src_feats['keypoints'] = torch.add(src_feats['keypoints'], src_pad_tensor)
            src_feats['descriptors'] = src_feats['descriptors'].unsqueeze(0)
            src_feats['keypoint_scores']= src_feats['keypoint_scores']
            dst_feats = extractor.extract(dst_image_tensor)
            dst_feats['scores'] = dst_feats.get('scores', torch.ones((1, dst_feats['keypoints'].shape[1]), device=dst_feats['keypoints'].device))
            dst_feats = rbd(dst_feats)
            dst_feats['keypoints'] = dst_feats['keypoints'].unsqueeze(0)
            #Now we need to transform the keypoint coordinates to the full panorama coordinates
            dst_pad_tensor = torch.from_numpy(np.tile(dst_pad_array, (dst_feats['keypoints'].shape[1], 1))).to(config["device"])
            dst_feats['keypoints'] = torch.add(dst_feats['keypoints'], dst_pad_tensor)
            dst_feats['descriptors'] = dst_feats['descriptors'].unsqueeze(0)
            dst_feats['keypoint_scores'] = dst_feats['keypoint_scores']
        img_feats[i]['src'] = src_feats
        img_feats[i]['dst'] = dst_feats
    return img_feats

def match_panorama_features(images, search_distance, config):
    #####################################################################
    #Extract panorama features and then match keypoints across panoramas#
    #####################################################################
    print('Extracting panorama features...')
    img_feats = extract_all_panorama_features(images, search_distance, config)
    img_match_dict = {} #Dictionary to store matching keypoints and features
    for i in range(len(images)):
        img_match_dict[i] = {'keypoints': {'src': [], 'dst': []}, 'features': {'src': [], 'dst': []}}
    #Use LightGlue for matching
    matcher = LightGlue(features="superpoint").eval().to(config["device"])
    print('Matching panorama features...')
    for i in range(len(images) - 1):
        feat_dict = {'image0': img_feats[i]['src'], 'image1': img_feats[i + 1]['dst']}
        img_matches = matcher(feat_dict)
        feats0, feats1, img_matches = [rbd(x) for x in [feat_dict['image0'], feat_dict['image1'], img_matches]]
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], img_matches["matches"]
        ###################################################
        #Find which keypoints were matched and move to cpu#
        ###################################################
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        m_kpts0_np = m_kpts0.cpu().numpy()
        m_kpts1_np = m_kpts1.cpu().numpy()
        ###############################################################################
        #Filter the matches based on RANSAC inliers of a partial affine transformation#
        ###############################################################################
        H, mask = cv2.estimateAffinePartial2D(m_kpts0_np, m_kpts1_np, cv2.RANSAC, ransacReprojThreshold  = 3.0)
        k0_idx = matches[:,0].cpu().numpy()[mask.astype(bool).flatten()]
        k1_idx = matches[:,1].cpu().numpy()[mask.astype(bool).flatten()]
        preselect_kp0, preselect_feat0 = kpts0[k0_idx].cpu().numpy(), feats0['descriptors'][k0_idx].cpu().numpy()
        preselect_kp1, preselect_feat1 = kpts1[k1_idx].cpu().numpy(), feats1['descriptors'][k1_idx].cpu().numpy()
        ########################
        #Store filtered matches#
        ########################
        img_match_dict[i]['keypoints']['src'] = [keypoint for keypoint in preselect_kp0.tolist()]
        img_match_dict[i]['features']['src'] = [feature for feature in preselect_feat0.tolist()]
        img_match_dict[i + 1]['keypoints']['dst'] = [keypoint for keypoint in preselect_kp1.tolist()]
        img_match_dict[i + 1]['features']['dst'] = [feature for feature in preselect_feat1.tolist()]
    return img_match_dict

def build_panorama_opencv_objects(images, img_matches):
    ########################################################
    #To provide access to OpenCV functions, convert matches#
    #into OpenCV objects for potential future use          #
    ########################################################
    cv_features = []
    for idx in range(len(images)):
        #Unpack matched features and keypoints
        feat = cv2.detail.computeImageFeatures2(cv2.ORB.create(), images[idx])
        keypoints = np.array(img_matches[idx]['keypoints']['src'] + img_matches[idx]['keypoints']['dst'])
        feat.keypoints = tuple(cv2.KeyPoint(keypoints[x, 0], keypoints[x, 1], 0.0) for x in range(len(keypoints)))
        feat.descriptors = cv2.UMat(np.array(img_matches[idx]['features']['src'] + img_matches[idx]['features']['dst'], dtype = np.float32))
        cv_features.append(feat)
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
            matcher = cv2.detail_AffineBestOf2NearestMatcher(full_affine = False, try_use_gpu = False, match_conf = 0.1, num_matches_thresh1 = 6)
            #apply2 finds all pairwise matches and is accelerated by TBB, but we can beat that performance
            #serially by simply skipping most pairs
            match = matcher.apply(cv_features[i], cv_features[j])
            match.src_img_idx, match.dst_img_idx = i, j
        matches.append(match)
    ################################################################################################
    #If not using OpenCV for the warping and camera estimation, we only need the matching keypoints#
    ################################################################################################
    img_keypoints = {i: img_matches[i]['keypoints'] for i in range(len(images))}
    return cv_features, matches, img_keypoints

def adjust_panoramas(batch_imgs, config):
    ##############################################################
    #Pad images to have equal size in the non-stitching direction#
    ##############################################################
    image_dims = np.array([img.shape for img in batch_imgs])[:,:2]
    #Put padding on the bottom of images
    pad = np.max(image_dims[:,0]) - image_dims[:,0]
    padded_images = [cv2.copyMakeBorder(batch_imgs[i], 0, pad[i],  0, 0, cv2.BORDER_CONSTANT) for i in range(len(batch_imgs))]
    return padded_images
    
def find_pano_corners(thresh, min_length, config):
    #######################################################################
    #To find the corners of the panorama, we find the convex hull points  #
    #of the thresholded image that are closest to the corners of the image#
    #######################################################################
    closest_corners = []
    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_pts = contours[0].reshape(contours[0].shape[0], contours[0].shape[2])
    hull = cv2.convexHull(cnt_pts)
    hull_pts = hull.reshape(hull.shape[0], hull.shape[2])
    ymax, xmax = thresh.shape
    top_left = [0, 0]
    top_right = [xmax, 0]
    bottom_left = [0, ymax]
    bottom_right = [xmax, ymax]
    all_corners = (top_left, top_right, bottom_right, bottom_left)
    for img_corner in all_corners:
        closest = np.sqrt(((hull_pts - img_corner)**2).sum(axis = 1))
        closest_corners.append(hull_pts[np.argmin(closest)])
    if closest_corners [2][1] - closest_corners[0][1] < min_length or closest_corners[3][1] -  closest_corners[1][1] < min_length:
        raise ValueError("Panorama corners were estimated poorly")
    return closest_corners

def warp_panorama(img, thresh, corners, spline_pts, config):
    ###########################################################################
    #Slice the panorama into chunks and then warp those chunks into rectangles# 
    #so we can recover a straightened panorama                                #
    ###########################################################################
    #Use splines to estimate the top and bottom edges of the panorama for smoother
    #slicing
    top_spline, bottom_spline = make_border_splines(thresh, corners, spline_pts)
    #Now we find the length of the vectors that slice the panorama vertically
    #and recalculate the end points so that the vectors are of equal length
    top_bottom_distances = bottom_spline - top_spline
    top_bottom_distances = np.sqrt((top_bottom_distances**2).sum(axis = 1))
    #Calculate the minimum vector length, we choose the median here to protect
    #against low minimum values due to poor corner attribution
    min_distances = np.median(top_bottom_distances)//2
    # height_ratio = top_bottom_distances/(2*min_distances)
    mid_points = np.mean([bottom_spline, top_spline], axis = 0)
    ###########################################
    #Estimate straightened panorama dimensions#
    ###########################################
    #We calculate the width of the new panorama as the length of the line
    #connecting all of the midpoints and the height to be the length of the 
    #vertical slices
    smd = np.sqrt(((mid_points[1:, :] - mid_points[:-1, :])**2).sum(axis = 1)) #Distances between mid points
    smd = np.round(smd).astype(int)
    height = int(2*min_distances)
    #We take the width of each slice as the distance between mid points and assume that the scale 
    #is already equalized from the original stitching process
    widths = smd 
    cumulative_widths = np.append(np.array([0]), np.cumsum(widths)).astype(int)
    #############################
    #Build straightened panorama#
    #############################
    #A mask to isolate each slice
    mask = np.zeros((thresh.shape[0], thresh.shape[1]), np.uint8)
    #Blank canvas where we will write our warped slices, we over allocate in size 
    blank = np.zeros((height, np.sum(widths), 3))
    for i in range(len(top_spline) - 1):
        #Corners of the sliced quadrilateral with top left, top right, bottom right, bottom left
        slice_corners = np.round(np.vstack((bottom_spline[i: i+2, :], np.flip(top_spline[i: i + 2,:], axis = 0))))
        warped = warp_slice(img, mask, slice_corners, widths[i], height)
        blank[:,cumulative_widths[i]: cumulative_widths[i + 1]] = warped
    blank = np.flip(blank.astype(np.uint8), axis = 0) #adjust array to go back to OpenCV image coordinates (0 on top)
    return blank.astype(np.uint8)

def make_border_splines(thresh, corners, spline_pts):
    ###############################################################
    #Generate points along the top and bottom border of a panorama#
    ###############################################################
    top_left, top_right, bottom_right, bottom_left = corners
    ################################################################
    #Move along the long axis and get the highest and lowest points#
    #in the thresholded image                                      #
    ################################################################
    top_contour_pts = []
    for x in range(top_left[0], top_right[0]):
        top = np.where(thresh[:, x] > 0)[0]
        if len(top) > 0:
            top_contour_pts.append([x, np.min(top)])
    top_contour_pts = np.array(top_contour_pts)
    bottom_contour_pts = []
    for x in range(bottom_left[0], bottom_right[0]):
        bottom = np.where(thresh[:, x] > 0)[0]
        if len(bottom) > 0:
            bottom_contour_pts.append([x, np.max(bottom)])
    bottom_contour_pts = np.array(bottom_contour_pts)
    ########################################################
    #Convert points into curves parameterized by arc length#
    #then rebuild splines with evenly spaced points        #
    ########################################################
    #Top contour
    dp = top_contour_pts[1:, :] - top_contour_pts[:-1, :]      
    l = (dp**2).sum(axis=1)       
    top_vec = np.sqrt(l).cumsum()   
    top_vec = np.r_[0, top_vec]
    spl = sp.interpolate.make_interp_spline(top_vec, top_contour_pts, axis=0)
    uu = np.linspace(top_vec[0], top_vec[-1], spline_pts)
    top_spline = spl(uu)
    #Bottom contour
    dp = bottom_contour_pts[1:, :] - bottom_contour_pts[:-1, :]     
    l = (dp**2).sum(axis=1)        
    bottom_vec = np.sqrt(l).cumsum()   
    bottom_vec = np.r_[0, bottom_vec]     
    spl = sp.interpolate.make_interp_spline(bottom_vec, bottom_contour_pts, axis=0)
    uu = np.linspace(bottom_vec[0], bottom_vec[-1], spline_pts)
    bottom_spline = spl(uu)
    return top_spline, bottom_spline

def warp_slice(img, mask, slice_corners, width, height):
    ################################################
    #For a given quadrilateral slice of a panorama,#
    #warp into a rectangle                         #
    ################################################
    #Mask everything but the slice
    rect_mask = cv2.fillConvexPoly(mask, slice_corners.reshape((-1, 1, 2)).astype(np.int32), 255)
    rect_img = cv2.bitwise_and(img, img, mask=rect_mask)
    #Resize the slice so it is in its in a rectangular bounding box
    rect_mask_corners = np.where(rect_mask == 255)
    maxx, maxy, minx, miny = np.max(rect_mask_corners[0]), np.max(rect_mask_corners[1]), np.min(rect_mask_corners[0]),  np.min(rect_mask_corners[1])
    rect_mask_bb = rect_img[minx:maxx, miny:maxy]
    #Get target points for a normal rectangle
    rect_points = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    #Translate the corners for a bounding box with top left corner at 0, 0
    translated_corners = np.copy(slice_corners)
    translated_corners[:, 0] -= miny
    translated_corners[:, 1] -= minx
    #Get perspective transform from slice to a rectangle and warp the slice
    H = cv2.getPerspectiveTransform(translated_corners.astype(np.float32), rect_points.astype(np.float32))
    warped = cv2.warpPerspective(rect_mask_bb, H, (width, height))
    return warped

def straighten_pano(img, width, min_length, config):
    #############################################################
    #Straighten panorama by cutting it into quadrilateral slices#
    #and projecting them to rectangles                          #
    #############################################################
    #Threshold the image to make a mask of the panorama
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(bw, 0, 255, 0) #Separate black from non-black pixels
    ######################################
    #Get panorama corners and slice width#
    ######################################
    #Slice the panorama  so the width of each slice is roughly 1.5x
    #the stitching dimension of a single image used in the panorama
    spline_pts = int(thresh.shape[1]//width) #width = search distance
    if spline_pts < 3:
        spline_pts = 3
    #Get panorama corners based on the vertical dimension of the panorama
    #being at least min_length, where min_length = 0.5 height of original images
    all_corners = find_pano_corners(thresh, min_length, config)
    ##########################
    #Slice and warp panorama#
    #########################
    rectangle_image = warp_panorama(img, thresh, all_corners, spline_pts, config)
    return rectangle_image

def match_pano_scale(straightened_imgs, search_distance, config):
    scaled_imgs = [straightened_imgs[0]]
    for i in range(len(straightened_imgs) - 1):
        src_ht, _ = get_stitch_edge_heights(scaled_imgs[i], search_distance, config)
        _, dst_ht = get_stitch_edge_heights(straightened_imgs[i + 1], search_distance, config)
        scale_factor = src_ht/dst_ht
        scaled = cv2.resize(straightened_imgs[i + 1], dsize = None, fx = scale_factor, fy = scale_factor)
        scaled_imgs.append(scaled)
    return scaled_imgs

def get_stitch_edge_heights(img, search_distance, config):
    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(bw, 0, 255, 0) #separate black from non-black pixels
    thresh = (thresh/255) 
    heights = np.sum(thresh, axis = 0)
    #We get the height at the 1/5 the width of the original image width since small angles in warping
    #can lead to edge effects
    test_position = int(search_distance * 0.2)
    src_stitch_height = heights[test_position]
    dst_stitch_height = heights[-test_position]
    return src_stitch_height, dst_stitch_height

def crop_panorama(panorama, config):
    ##########################################################
    #This function will remove black space, but assumes that #
    #the panorama is already relatively straight.            #
    ##########################################################
    bw = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(bw, 0, 255, 0) #separate black from non-black pixels
    height = thresh.shape[0]
    sums = np.sum(thresh, axis = 1)
    max_sum = np.max(sums)
    good_idx = np.where(sums >= 0.95*max_sum)[0]
    start = np.min(good_idx)
    stop = np.max(good_idx)
    cropped = panorama[start:stop, :]

    #######################################################
    #Only crop if most of the panorama height is preserved#
    #######################################################
    if stop - start > 0.5*height:
        return cropped
    else:
        print("Could not crop panorama because it is not straight. Consider loweing the batch size.")
        return panorama
    
def batch_only(config_path):
    ##############################################
    #If the panoramas are already generated,     #
    #generate the super panorama from saved files#
    ##############################################
    config = load_config(config_path) #Load config
    #We assume that the only files in the batch_path are the panoramas
    parent_dir = os.path.dirname(os.path.normpath(config["image_directory"]))
    output_dir = os.path.basename(os.path.normpath(config["image_directory"])) + '_output'
    output_path = os.path.join(parent_dir, output_dir)
    batch_paths = [os.path.join(output_path, img_name) for img_name in os.listdir(output_path)]
    #We sort the panoramas by their image id
    num = np.array([int((p.split("\\")[-1]).split("_")[0]) for p in batch_paths])
    sorted_indices = np.argsort(num)
    sorted_batch_paths = [batch_paths[i] for i in sorted_indices]
    #Then run the super stitcher
    stitch_super_panorama(sorted_batch_paths, config)
    
def stitch_super_panorama(batch_paths, config):
    #######################################################################
    #Take existing panoramas and stitch them into one final super panorama#
    #######################################################################
    print('Disabling OpenCL to try to avoid memory issues')
    cv2.ocl.setUseOpenCL(False)
    print("Retrieving panoramas")
    ##################################################
    #Read in the previously created panoramas and pad#
    ##################################################
    batch_imgs = [cv2.imread(batch_path) for batch_path in batch_paths]
    #Create a uniform border around the panoramas to aid in straightening the images
    batch_imgs = [cv2.copyMakeBorder(batch_img, 100, 100, 100, 100, cv2.BORDER_CONSTANT) for batch_img in batch_imgs]
    #############################################################
    #Recover the original image dimensions used for the panorama#
    #############################################################
    dummy_path = config["image_directory"]
    image_paths = [os.path.join(dummy_path, img_name) for img_name in os.listdir(dummy_path)]
    image = cv2.imread(image_paths[0]) #load first image to get dimensions
    dummy_img = cv2.resize(image, dsize = None, fx = config["final_resolution"], fy = config["final_resolution"])
    if config["stitching_direction"] == 'RIGHT':
        #Flip image across vertical axis so the stitching edge is now on the left
        dummy_img = cv2.flip(dummy_img, 1)
    elif config["stitching_direction"] == 'UP':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        dummy_img = cv2.rotate(dummy_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif config["stitching_direction"] == 'DOWN':
        #Rotate 90 degrees CCW so stitching edge is now on the left
        dummy_img = cv2.rotate(dummy_img, cv2.ROTATE_90_CLOCKWISE)
    img_xdim, img_ydim = dummy_img.shape[1], dummy_img.shape[0]
    img_dims = (img_xdim, img_ydim)
    ######################################################################
    #Specify search distance and minimum expected height of each panorama#
    ######################################################################
    #To restrict feature matching to the stitching ends of the panoramas, we only search for 
    #features in a search distance from the stitching ends.
    #Since subsequent panoramas overlap by one image, the relevant features should be within one
    #image width of the ends, but since these images may have a different scale or be rotated, we 
    #search at 1.0x the original image width.
    #When finding the corners of each panorama, we use the height of the original images to help 
    #rule out edges that are too short, here we assume that the corners should be longer than half the
    #original image height
    search_distance = int(img_dims[0] *1.0)
    length = img_dims[1]*0.5

    #################################
    #Straighten and resize panoramas#
    #################################
    #Straighten the panoramas to make it easier to stitch them into a super panorama
    #because the long length of the panoramas means that any rotation when matching
    #panoramas will lead to a large displacement at the other end of the panorama.
    #We assume that the camera path is roughly linear and that panoramas should only be offset to properly stitch them,
    #but to assume that the no rotation is necessary to stitch the panoramas, we should ensure that the edges of the panorama
    #are normal to the camera movement. Since deviations of the camera orientation from the normal plane will cause curving
    #in the panoramas, we pre-straighten the panoramas to make the super panorama stitching easier
    print("Straightening panoramas...")
    adjusted_imgs = [straighten_pano(bordered_img, search_distance, length, config) for bordered_img in batch_imgs]
    #Match scale across panoramas
    adjusted_imgs = match_pano_scale(adjusted_imgs, search_distance, config)
    #Now pad the straightened images so the images are all of the same dimension in the non-stitching direction
    adjusted_imgs = adjust_panoramas(adjusted_imgs, config)

    ########################################################################
    #Find features and matches between the ends of subsequent panoramas and#
    #then stitch them together without any rotation                        #
    ########################################################################
    img_matches = match_panorama_features(adjusted_imgs, search_distance, config)
    cv_features, matches, img_keypoints = build_panorama_opencv_objects(adjusted_imgs, img_matches)
    print("Stitching panoramas...")
    super_panorama = affine_OpenCV_pipeline(adjusted_imgs, img_keypoints, True, config)
    ###########################################################
    #Crop image to center the height of the panorama such that#
    #the cropped image has hardly any black space             #
    ###########################################################
    if config["crop"]:
        super_panorama = crop_panorama(super_panorama, config)
    ###########################
    #Save final super panorama#
    ###########################
    print('Saving final panorama ...')
    parent_dir = os.path.dirname(os.path.normpath(config["image_directory"]))
    output_dir = os.path.basename(os.path.normpath(config["image_directory"])) + '_output'
    output_path = os.path.join(parent_dir, output_dir)
    output_filename = 'batch_' + os.path.basename(os.path.normpath(config["image_directory"])) + '.png'
    cv2.imwrite(os.path.join(output_path, "straightened_super_"  + output_filename), super_panorama)
    
def run_batches(config_path):
    ###################
    #Run full pipeline#
    ###################
    config = load_config(config_path) #Load config
    parent_dir = os.path.dirname(os.path.normpath(config["image_directory"]))
    output_dir = os.path.basename(os.path.normpath(config["image_directory"])) + '_output'
    output_path = os.path.join(parent_dir, output_dir)
    output_filename = 'batch_' + os.path.basename(os.path.normpath(config["image_directory"])) + '.png'
    if not os.path.exists(output_path):
        print("Creating ", output_path)
        os.makedirs(output_path)
    start_idx = 0 #The image to start stitching
    finished = False #True when you stitch the final image in the directory
    #Stores the img idx of the final image used in the batch and whether the stitching was successful
    batch_dict = {} 
    ##############################################################
    #Working in batches of images, first construct mini panoramas#
    ##############################################################
    while not finished:
        #Create panoramas by first stitching batches of images
        #and starting a new stitch from the last image of the previous
        #panorama, leaving one image of overlap between each subsequent panorama
        finished, start_idx, success = run_stitching_pipeline(start_idx, config)
        batch_dict[start_idx] = success
    ###########################################################
    #Determine if all of the images were stitched successfully#
    ###########################################################
    for key, outcome in batch_dict.items():
        if outcome == False:
            print(key, " panorama failed")
    ######################################################################
    #If all panorama batches were successful, attempt to stitch them into#
    #the final super panorama.                                           #
    ######################################################################
    if np.all(list(batch_dict.values())):
        #Retrieve the batches of panoramas that were created
        batch_paths = [os.path.join(output_path, str(i) + "_" + output_filename) for i in list(batch_dict.keys())]
        #If there is more than one panorama, stitch the panoramas together
        if len(batch_paths) > 1:
            stitch_super_panorama(batch_paths, config)
        #Otherwise, save the single panorama
        else:
            if config["crop"]:
                super_panorama = crop_panorama(batch_paths[0], config)
                print('Saving final panorama ...')
                cv2.imwrite(os.path.join(output_path, "super_"  + output_filename), super_panorama)
    else:
        print("Could not proceed due to failed panoramas")

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

if __name__ == "__main__":
    config_path = sys.argv[1]
    run_batches(config_path)
    # batch_only(config_path)