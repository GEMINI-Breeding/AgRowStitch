# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:33:41 2024

@author: Kaz Uyehara
"""
import cv2
from lightglue import LightGlue, SuperPoint #git clone https://github.com/cvg/LightGlue.git && cd LightGlue
from lightglue.utils import rbd
import itertools
import multiprocessing
import numpy as np
import os
from stitching.blender import Blender
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_estimator import CameraEstimator
from stitching.camera_wave_corrector import WaveCorrector
from stitching.cropper import Cropper
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.seam_finder import SeamFinder
from stitching.warper import Warper
import sys
import torch
from torchvision import transforms
import yaml


def extract_features(img_path, config):
    ##################################
    #Load image and reduce resolution#
    ##################################
    image = cv2.imread(img_path) #only accepts multiple images
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
        

def get_inliers(img_feats, src_idx, dst_idx, src_pixel_limit, dst_pixel_limit, config):
    ###########################################################
    #Use LightGlue to match features extracted from SuperPoint#
    ###########################################################
    matcher = LightGlue(features="superpoint").eval().to(config["device"])
    feat_dict = {'image' + str(i): img_feats[j] for i, j in enumerate([src_idx, dst_idx])}
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
    if config["stitching_direction"] == 'LEFT':
        filtered_idx = np.where((m_kpts0_np[:,0] < src_pixel_limit) & (m_kpts1_np[:,0] > dst_pixel_limit))
    elif config["stitching_direction"] == 'UP':
        filtered_idx = np.where((m_kpts0_np[:,1] < src_pixel_limit) & (m_kpts1_np[:,1] > dst_pixel_limit))
    elif config["stitching_direction"] == 'RIGHT':
        filtered_idx = np.where((m_kpts0_np[:,0] > src_pixel_limit) & (m_kpts1_np[:,0] < dst_pixel_limit))
    else:
        filtered_idx = np.where((m_kpts0_np[:,1] < src_pixel_limit) & (m_kpts1_np[:,1] > dst_pixel_limit))
        
    #We want src points to be close to dst points in global space, so we exclude points that are
    #on the wrong part of the image to try to get the minimal images necessary and keypoints that are
    #unlikely to be present in more than two images.
    m_kpts0_np = m_kpts0_np[filtered_idx]
    m_kpts1_np = m_kpts1_np[filtered_idx]

    ###############################################################
    #Filter out matches that do not fit our homography constraints#
    ###############################################################
    if len(m_kpts0_np) >= config["min_inliers"]:
        #Changing the RANSAC threshold parameter will determine if we get more but noisier matches (higher value)
        #or fewer but more pixel-perfect matches (lower value). Lower values help ensure that the OpenCV Matcher
        #will also match the points.
        transformation_matrix, mask = cv2.findHomography(m_kpts0_np, m_kpts1_np, cv2.RANSAC, 2.0)
        if transformation_matrix is None:
            return (False, None, None, None, None)
        else:
            #We assume that the 
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
                (abs(scale - 1.0) < config["scale_constraint"] and np.sum(mask) >= config["min_inliers"])):
                # confidence = np.sum(mask) / (8 + 0.3 * mask.size)
                k0_idx = matches[:,0].cpu().numpy()[filtered_idx][mask.astype(bool).flatten()]
                k1_idx = matches[:,1].cpu().numpy()[filtered_idx][mask.astype(bool).flatten()]
                preselect_kp0, preselect_feat0 = kpts0[k0_idx].cpu().numpy(), feats0['descriptors'][k0_idx].cpu().numpy()
                preselect_kp1, preselect_feat1 = kpts1[k1_idx].cpu().numpy(), feats1['descriptors'][k1_idx].cpu().numpy()
                return (True, preselect_kp0, preselect_feat0, preselect_kp1, preselect_feat1)
            else:
                return (False, None, None, None, None)
    else:
        return (False, None, None, None, None)

def check_forward_matches(img_dict, src_idx, img_feats, img_dim, config):
    ######################################################################
    #Try to find matches with keypoints clustered on their stitching edge#
    ######################################################################
    keypoint_prop = config["keypoint_prop"]
    
    while keypoint_prop <= 1.0: #the side of the image 
        #We prefer the images to have keypoints close to their stitching edge so that keypoints are
        #unlikely to appear more than twice and we can minimize the number of images used.
        #The keypoint prop is the part of the image we expect to find keypoints, e.g. if 
        #keypoint prop is 0.5, we want the keypoints to be on the correct half of both images -- 
        #closer to the stitching edge than the non-stitching edge
        
        ##############################################
        #Check  images starting with far images first#
        ###############################################
        for dst_idx in range(src_idx + config["forward_limit"], src_idx, -1):
            #Get correct values depending on stitching direction
            if config["stitching_direction"] == 'LEFT' or config["stitching_direction"] == 'UP':
                src_pixel_limit = img_dim*(keypoint_prop)
                dst_pixel_limit = img_dim*(1 - keypoint_prop)
            else:
                src_pixel_limit = img_dim*(1 - keypoint_prop)
                dst_pixel_limit = img_dim*(keypoint_prop)
                
            if len(img_feats) > dst_idx:
                ###################################
                #Match images based on constraints#
                ###################################
                matched, kps, fs, kpd, fd = get_inliers(img_feats, src_idx, dst_idx, src_pixel_limit, dst_pixel_limit, config)

                ###############################################
                #Save keypoints and features for use in OpenCV#
                ###############################################
                if matched:
                    img_dict[src_idx]['keypoints'] = img_dict[src_idx]['keypoints'] + [
                                                    keypoint for keypoint in kps.tolist()
                                                    if keypoint not in img_dict[src_idx]['keypoints']]
                    img_dict[src_idx]['features'] = img_dict[src_idx]['features'] + [
                                                    feature for feature in fs.tolist()
                                                    if feature not in img_dict[src_idx]['features']]
                    img_dict[dst_idx]['keypoints'] = img_dict[dst_idx]['keypoints'] + [
                                                    keypoint for keypoint in kpd.tolist()
                                                    if keypoint not in img_dict[dst_idx]['keypoints']]
                    img_dict[dst_idx]['features'] = img_dict[dst_idx]['features'] + [
                                                    feature for feature in fd.tolist()
                                                    if feature not in img_dict[dst_idx]['features']]
                    print('Matched ', src_idx, dst_idx, ' with proportion limit ', keypoint_prop, ' and ', len(kps), ' inliers')
                    return img_dict, dst_idx
        
        #If we cannot fina matches within constraints, loosen the position of the keypoints and try again
        keypoint_prop += 0.1 
        
    ###################
    #Cannot find match#
    ###################
    """TO DO"""
    #If no matches can be found at all -- should add a save function that will continue
    #with current batch of images (until src_idx) and run the pipeline, then start again 
    #and try with image src_idx + forward_limit
    raise ValueError("Could not find a match with ", src_idx, " try lowing min_inliers or increasing forward_limit ")



def get_all_features(img_paths, config):
    ###################################################
    #Extract features from all images using SuperPoint#
    ###################################################
    image = cv2.imread(img_paths[0]) #load first image to get dimensions
    dummy_img = cv2.resize(image, dsize = None, fx = config["feature_resolution"], fy = config["feature_resolution"])
    img_xdim, img_ydim = dummy_img.shape[1], dummy_img.shape[0]

    #Need to know which part of the image we should expect src points to be on vs. dst points
    if config["stitching_direction"] == 'LEFT' or config["stitching_direction"] == 'RIGHT':
        img_dim = img_xdim
    else:
        img_dim = img_ydim
    
    print('Extracting features...')
    img_feats = [extract_features(img, config) for img in img_paths]

    return img_feats, img_dim

def find_matching_images(img_feats, img_dim, config):
    ##########################################################
    #Using features extracted from SuperPoint, use LightGlue #
    #to find the minimum set of images that can be stitched  #
    # at high confidence and connect the first and last image#
    ##########################################################
    img_dict = {}
    for i in range(len(img_feats)):
        img_dict[i] = {'keypoints': [], 'features': []}
        
    src_idx, dst_idx = 0, 0
    image_subset = [src_idx]
    
    print('Finding best image matches...')
    while src_idx < len(img_feats) - 1:
        #only add keypoints and features to the dictionary if they are the best match for that image
        img_dict, dst_idx = check_forward_matches(img_dict, src_idx, img_feats, img_dim, config)
        src_idx = dst_idx
        image_subset.append(dst_idx)
        
    print('Using ', len(image_subset), ' images of the initial ', len(img_feats))
    print('Image subset: ', image_subset)
    return img_dict, image_subset

def build_feature_objects(subset_image_paths, img_dict, subset_indices, config):
    ##############################################################################
    #Convert the preselected SuperPoint keypoint and features into OpenCV objects#
    ##############################################################################
    cv_features = []
    image = cv2.imread(subset_image_paths[0]) #load first image to get dimensions
    dummy_img = cv2.resize(image, dsize = None, fx = config["feature_resolution"], fy = config["feature_resolution"])
    #We pass the real images here, but since we set the attributes, this can be replaced by a dummy image
    for idx in subset_indices:
        feat = cv2.detail.computeImageFeatures2(cv2.ORB.create(), dummy_img)
        keypoints = np.array(img_dict[idx]['keypoints'])
        feat.keypoints = tuple(cv2.KeyPoint(keypoints[x, 0], keypoints[x, 1], 0.0) for x in range(len(keypoints)))
        feat.descriptors = cv2.UMat(np.array(img_dict[idx]['features'], dtype = np.float32))
        cv_features.append(feat)
    
    return cv_features

def subset_images(image_paths, config):
    ##############
    #Get features#
    ##############
    img_feats, img_dim = get_all_features(image_paths, config)
    
    ##################################
    #Find best matches between images#
    ##################################
    img_dict, subset_indices = find_matching_images(img_feats, img_dim, config)

    ###########################################################################
    #Use the matched images and keypoints to create the OpenCV feature objects#
    ###########################################################################
    subset_image_paths = [image_paths[i] for i in subset_indices]
    cv_features = build_feature_objects(subset_image_paths, img_dict, subset_indices, config)
    
    return cv_features, subset_indices

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
    cv_features, subset_indices = subset_images(image_paths, config)

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
            #considered a match, default is 3.0
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

    return images, cv_features, matches

    
def spherical_OpenCV_pipeline(images, features, matches, config):
    #####################################################################
    #Process images assuming that the camera is stationary and rotating.#
    #Use when 
    ###################################################
    cameras = spherical_camera_estimation(features, matches, config)
    processed_images = spherical_warp_images(images, cameras, config)
    processed_images = crop_images(*processed_images, images, config)
    processed_images = get_seams_and_compensate(*processed_images)
    panorama = blend_images(*processed_images)
    return panorama

def affine_OpenCV_pipeline(images, features, matches, config):
    ##########################################################
    #Process images assuming that they are on a single plane.#
    #Use when the camera is moving in time.                  #
    ##########################################################
    cameras = affine_camera_adjustment(features, matches)
    processed_images = affine_warp_images(images, cameras, config)
    processed_images = crop_images(*processed_images, images, config)
    processed_images = get_seams_and_compensate(*processed_images)
    panorama = blend_images(*processed_images)
    return panorama

def affine_camera_adjustment(features, matches):
    ###############################################################
    #Estimate and adjust homography matrices in global coordinates#
    ###############################################################
    print('Estimating cameras...')
    estimator = cv2.detail_AffineBasedEstimator()
    success, cameras = estimator.apply(features, matches, None)
    if not success:
        raise ValueError("Failed to estimate cameras")
    #change types to match what bundleAdjuster wants
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
        
    print('Adjusting cameras...')
    #Changing confidence threshold may help adjustment if the optimization is difficult 
    #and fails. It should be easier if there is high confidence between subsequent images
    #and the correct adjustment (affine or spherical) is chosen. Lower the confidence threshold
    #to pass adjustment, but adjustment might be more error prone.
    adjuster = cv2.detail_BundleAdjusterAffine() #use cv2.detail_BundleAdjusterAffinePartial to remove shearing
    adjuster.setConfThresh(0.25)
    success, cameras =adjuster.apply(features, matches, cameras)
    if not success:
        raise ValueError("Failed to adjust cameras")
        
    return cameras

def affine_warper(original_img, camera, aspect_ratio):
    ############################################################################################
    #Given global homography matrix, apply homography on image in local coordinate system      #
    #to find the final size of its bounding rectangle and the global offsets of the top left   #
    #corner of that rectangle. Repeat for a mask of the image for easier processing downstream.#
    #This allows us to place the transformed image into the global coordinate system.          #
    ############################################################################################
    #Make rectangle corner coordinates to transform by homography matrix to know new dimensions of image
    w, h = original_img.shape[1], original_img.shape[0]
    x, y = 0, 0
    local_corners = np.array([[x, y, 1], [x, y + h - 1, 1], [x + w - 1, y, 1], [x + w - 1, y + h - 1, 1]])
    
    ###########################################################################################################
    #Invert homography matrix to go from local image to global coordinates, aspect ratio will scale image size#
    #to allow us to downscale images for low resolution versions for seams and cropping.                      #
    ###########################################################################################################
    inverted = np.linalg.inv(camera.R) * aspect_ratio
    top_left = np.floor(np.matmul(inverted, local_corners[0])[:2])
    bottom_left = np.floor(np.matmul(inverted, local_corners[1])[:2])
    top_right = np.floor(np.matmul(inverted, local_corners[2])[:2])
    bottom_right =  np.floor(np.matmul(inverted, local_corners[3])[:2])
    local_translationx, local_translationy = max(top_left[0] - bottom_left[0], 0), max(top_left[1] - top_right[1], 0)
    #Find rectangle that will inscribe transformed image
    x_corners = np.array([a[0] for a in [top_left, bottom_left, top_right, bottom_right]])
    y_corners = np.array([a[1] for a in [top_left, bottom_left, top_right, bottom_right]])
    width = int(np.ceil(np.max(x_corners) -  np.min(x_corners)))
    height = int(np.ceil(np.max(y_corners) -  np.min(y_corners)))

    inverted[:,2] = np.array([0, 0, 1]) #Remove global translation
    #Add translation so that the rotated image fits into the canvas instead of starting in top left corner
    T = np.array([[1, 0, local_translationx], [0, 1, local_translationy], [0, 0, 1]])
    H_local = T.dot(inverted)
    warped = cv2.warpPerspective(original_img, H_local, (width, height), cv2.INTER_LINEAR)
    mask = mask = 255 * np.ones((h, w), np.uint8)
    mask = cv2.warpPerspective(mask, H_local, (width, height), cv2.INTER_NEAREST)
    return warped, mask, tuple(top_left.astype(int)), (width, height)

def affine_warp_images(images, cameras, config):
    ##################################################################################################
    #Get the transformed images and their masks as well as their start corners in global coordinates.#
    #Repeat for low resolution images so processing is easier downstream                             #
    ##################################################################################################
    print('Warping images...')
    warped_final_imgs = []
    warped_final_masks = []
    final_corners = []
    final_sizes = []
    for img, camera in zip(images, cameras):
        warped_img, warped_mask, corner, size = affine_warper(img, camera, 1)
        warped_final_imgs.append(warped_img)
        warped_final_masks.append(warped_mask)
        final_corners.append(corner)
        final_sizes.append(size)
        
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

def spherical_camera_estimation(features, matches, config):
    ###########################################################################
    #Estimate camera rotations and focal length (can change across cameras),  #
    #with principalx and principaly constant across cameras and no translation#
    ###########################################################################
    print('Estimating cameras...')
    camera_estimator = CameraEstimator(estimator = 'homography')
    cameras = camera_estimator.estimate(features, matches)
    print('Adjusting cameras...')
    #Changing confidence threshold may help adjustment if the optimization is difficult 
    #and fails. It should be easier if there is high confidence between subsequent images
    #and the correct adjustment (affine or spherical) is chosen. Lower the confidence threshold
    #to pass adjustment, but adjustment might be more error prone.
    camera_adjuster = CameraAdjuster(adjuster = "ray", confidence_threshold = 0.5)
    cameras = camera_adjuster.adjust(features, matches, cameras)
    #The wave corrector will help keep the horizon linear, but should
    #not be relevant for most applications
    wave_corrector = WaveCorrector()
    cameras = wave_corrector.correct(cameras)
    return cameras

def spherical_warp_images(images, cameras, config):
    ####################################################################
    #Project images onto a sphere, use when doing a wide angle panorama#
    ####################################################################
    print('Warping images...')
    warper = Warper(warper_type = "spherical")
    warper.set_scale(cameras)
    
    low_imgs = [cv2.resize(img, dsize = None, fx = config["seam_resolution"], fy = config["seam_resolution"]) for img in images]
    low_sizes = [(i.shape[1], i.shape[0]) for i in low_imgs]
    camera_aspect = config["seam_resolution"]  # since cameras were obtained on medium imgs
    
    warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
    warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
    low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)
    
    final_sizes = [(i.shape[1], i.shape[0]) for i in images]
    camera_aspect = config["final_resolution"]/config["feature_resolution"]
    warped_final_imgs = list(warper.warp_images(images, cameras, camera_aspect))
    warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
    final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

    return (warped_low_imgs, warped_low_masks, low_corners, low_sizes, warped_final_imgs, warped_final_masks, final_corners, final_sizes)

def crop_images(warped_low_imgs, warped_low_masks, low_corners, low_sizes,
                warped_final_imgs, warped_final_masks, final_corners, final_sizes, images, config):
    ################################################################
    #Crop images into rectangle based on largest interior rectangle#
    ################################################################
    if config["crop"]:
        print('Cropping images...')
    else:
        print('Not cropping images...')

    #Use low resolution images to find rectangle
    cropper = Cropper(crop = config["crop"])
    low_corners = cropper.get_zero_center_corners(low_corners)
    cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)
    
    cropped_low_masks = list(cropper.crop_images(warped_low_masks))
    cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
    low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)
    
    lir_aspect = config["seam_resolution"]  # since lir was obtained on low imgs
    cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
    cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
    final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

    return (cropped_low_imgs, cropped_low_masks, low_corners, low_sizes, cropped_final_imgs, cropped_final_masks, final_corners, final_sizes)

def get_seams_and_compensate(cropped_low_imgs, cropped_low_masks, low_corners, low_sizes,
                             cropped_final_imgs, cropped_final_masks, final_corners, final_sizes):
    #######################################################################################
    #Find seams in overlapping areas and compensate the images for differences in exposure#
    #######################################################################################
    
    print('Finding seams...')    
    seam_finder = SeamFinder()
    seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
    seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_final_masks)]
    print('Compensating images...')    
    compensator = ExposureErrorCompensator()
    compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)
    compensated_imgs = [compensator.apply(idx, corner, img, mask) 
                        for idx, (img, mask, corner) 
                        in enumerate(zip(cropped_final_imgs, cropped_final_masks, final_corners))]

    return seam_masks, compensated_imgs, final_corners, final_sizes

def blend_images(seam_masks, compensated_imgs, final_corners, final_sizes):
    ###################################
    #Blend images together using seams#
    ###################################
    print('Preparing to blend images...')        
    blender = Blender()
    blender.prepare(final_corners, final_sizes)
    for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
        blender.feed(img, mask, corner)
    print('Blending images...')
    panorama, _ = blender.blend()
    print('Finished stitching')
    return panorama
    
def load_config(config_path):
    ################################################################
    #Load configuration from a YAML file and compile regex patterns#
    ################################################################
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    # Adjust device setting based on the string from config
    config["device"] = torch.device(config["device"] if torch.cuda.is_available() and config["device"] == "cuda" else "cpu")
    if config['cpus'] > multiprocessing.cpu_count():
        config['cpus'] = multiprocessing.cpu_count()
    print("Configuration loaded successfully...")
    return config
    
def run_stitching_pipeline(config_path):
    config = load_config(config_path)
    images, cv_features, matches = prepare_OpenCV_objects(config)
    if config['projection'] == "spherical":
        try:
            panorama = spherical_OpenCV_pipeline(images, cv_features, matches, config)
        except (ValueError):
            print("Spherical projection failed, trying affine projection instead...")
            panorama = affine_OpenCV_pipeline(images, cv_features, matches, config)
    else:
        panorama = affine_OpenCV_pipeline(images, cv_features, matches, config)
    if config['save_image']:
        print('Saving image...')
        cv2.imwrite(os.path.join(config['output_dir'], config['output_filename']), panorama)
    

if __name__ == "__main__":
    config_path = sys.argv[1]
    run_stitching_pipeline(config_path)
