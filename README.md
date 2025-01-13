# Panorama-Maker
For stitching and georeferencing images using LightGlue and custom post-processing scripts

### Installation

To install:

```
pip install .
```

Additionally, this library depends on LightGlue which must be installed from github.

```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

## Running
Run "minimal_image_stitching_LG_OpenCV.py" from the command line with the path to your config.yaml as the only argument.

### Stitching images
Update the "config.yaml" to customize parameters and then pass the path of the config file when running the script. At a minimum, the image directory and output directories must be updated to run the pipeline. The images should contain a sequential number within the filename which corresponds to the timestamp or order in which the images should be stitched. We assume that subsequent images have overlapping areas, but we attempt to stitch only a subset of the images with the forward_limit corresponding to the number of images past the focal image that will be checked for stitching. As the overlap between subsequent images increases, the forward_limit should increase so that as few images as necessary are used. There is a balance between ensuring strong matching so that false matches are ignored and the pipeline failing to find any matches. The min_inliers and max_reprojection_error are the main parameter for changing the confidence between matches. We recommend extracting frames at high fps for more reliable stitching.

### Concept
Since we will use the stitched images to segment structures and extract other phenotypic information, it is important that plant structures not be duplicated or ommitted from the stitched image. However, stitched plant images are particular susceptible to poor stitching because: 1) feature extraction using traditional methods (e.g. ORB, SIFT) may lead to poor matching when the image is filled with similar objects (e.g. leaves); 2) plants are non-planar, making it difficult to find the correct homography between images; and 3) plants are not rigid, so there may be simultaneous camera movement and object movement.

To reduce the occurence of poor stitches, we primarily aim to reduce the number of images used in the final stitched image and filter features. We assume that images are ordered and that the overlap between images decreases for more distant image pairs, so we iteratively search for the most distant image (set by forward_limit) that can be reliably matched with the focal image. To determine whether images are sufficiently matched, we use LightGlue to extract and match features and then require that the images have a high number of matched keypoints (min_inliers) using RANSAC with a maximum threshold (max_RANSAC_thresh) and that those matched keypoints have a low reprojection error (max_reprojection_error). To eliminate potential keypoint mismatches, we exclude keypoints based on their position on the source and destination images (keypoint_prop) such that source keypoints and destination keypoints must be be close to their shared stitching edge. We reject the pair of images if the homography of the matches violates our assumptions about the camera movement (xy_ratio, scale_constraint). For matches that conform to the camera constraints, we try to further filter out false matches (refine) by iteratively removing keypoints with the highest reprojection error, with the assumption that those keypoints correspond to moving or non-planar features. If these constraints cannot be satisfied, we proceed with the original RANSAC inliers.

The resulting keypoint and feature matches are then passed into the OpenCV stitching pipeline while also forcing the images to be stitched in their given order. We utilize the OpenCV bundle adjustment (camera adjustment), seam finding (dynamic programming color gradient), and blending (multiband) to create the final panorama. However, the OpenCV pipeline assumes a stationary (no translation) camera rotating about its axis rather than a moving camera. As such, the images are projected onto a sphere rather than a plane and the degrees of freedom optimized in the bundle adjustment are camera rotations and focal length. We are still determining the extent to which this assumption can be relaxed, but the pipeline works for translating cameras moving mostly in one direction for short distances (e.g. 20-30 images), while too many images leads to bundle adjustment failure or unreasonable run times because the optimization problem becomes too difficult. For this reason, optional bundle adjustment that works on camera translation and rotation using estimates from homography should be implemented in a future version.
