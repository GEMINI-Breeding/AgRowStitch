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
Run "image_stitching.py" from the command line with the path to your config.yaml as the only argument.

### Stitching images
Update the "config.yaml" to customize parameters and then pass the path of the config file when running the script. At a minimum, the image directory and output directories must be updated to run the pipeline. The images should contain a sequential number within the filename which corresponds to the timestamp or order in which the images should be stitched. We assume that subsequent images have overlapping areas, but we attempt to stitch only a subset of the images with the forward_limit corresponding to the number of images past the focal image that will be checked for stitching. As the overlap between subsequent images increases, the forward_limit should increase so that as few images as necessary are used. There is a balance between ensuring strong matching so that false matches are ignored and the pipeline failing to find any matches. The min_inliers and max_reprojection_error are the main parameters for changing the confidence between matches. We recommend extracting frames at high fps and using a small batch size (e.g. 20) for more reliable stitching.

### Concept
Since we will use the stitched images to segment structures and extract other phenotypic information, it is important that plant structures not be duplicated or ommitted from the stitched image. However, stitched plant images are particular susceptible to poor stitching because: 1) feature extraction using traditional methods (e.g. ORB, SIFT) may lead to poor matching when the image is filled with similar objects (e.g. leaves); 2) plants are non-planar, making it difficult to find the correct homography between images; and 3) plants are not rigid, so there may be simultaneous camera movement and object movement.

To reduce the occurence of poor stitches, we primarily aim to reduce the number of images used in the final stitched image and filter features. We assume that images are ordered and that the overlap between images decreases for more distant image pairs, so we iteratively search for the most distant image (set by forward_limit) that can be reliably matched with the focal image. To determine whether images are sufficiently matched, we use LightGlue to extract and match features and then require that the images have a high number of matched keypoints (min_inliers) using RANSAC with a maximum threshold (max_RANSAC_thresh) and that those matched keypoints have a low reprojection error (max_reprojection_error). To eliminate potential keypoint mismatches, we exclude keypoints based on their position on the source and destination images (keypoint_prop) such that source keypoints and destination keypoints must be be close to their shared stitching edge. We reject the pair of images if the homography of the matches violates our assumptions about the camera movement (xy_ratio, scale_constraint). For matches that conform to the camera constraints, we try to further filter out false matches (refine) by iteratively removing keypoints with the highest reprojection error, with the assumption that those keypoints correspond to moving or non-planar features. If these constraints cannot be satisfied, we proceed with the original RANSAC inliers.

The resulting keypoint and feature matches are then passed into the OpenCV stitching pipeline while also forcing the images to be stitched in their given order. We utilize the OpenCV bundle adjustment (camera adjustment), seam finding (dynamic programming color gradient), and blending (multiband) to create the final panorama. However, the OpenCV pipeline assumes a stationary (no translation) camera rotating about its axis rather than a moving camera. As such, the images are projected onto a sphere rather than a plane and the degrees of freedom optimized in the bundle adjustment are camera rotations and focal length. We are still determining the extent to which this assumption can be relaxed, but the pipeline works for translating cameras moving mostly in one direction for short distances (e.g. 20-30 images), while too many images leads to bundle adjustment failure or unreasonable run times because the optimization problem becomes too difficult. A partial affine transformation is also available (camera) when the spherical projection assumption performs worse than a partial affine stitch without bundle adjustment. Both camera modes (spherical or partial_affine) will create panoramas that "drift" if the camera angle does not remain normal to the image plane, though this is partially corrected in the spherical projection using wave correction. Regardless of the camera mode used, plant movement or matching using very non-planar points can create "ghosting" even after seam finding, though the spherical camera mode performs better than the partial affine mode.

To accomodate panoramas that require many images (i.e. high camera translation), we stitch the panorama in batches by making a panorama every batch_size images stitched. If a panorama cannot be generated, the constraints are modified or the camera mode is changed to try to produce a successful panorama. These panoramas are saved to the output_dir and then loaded if these smaller panoramas have all been successfully generated. Since the panoramas may be quite large, stitching the panoramas together into a super panorama can be difficult because error in rotations will cause large displacements. To improve the stitching of the super panorama, we straighten and rescale each panorama prior to stitching and assume that all panoramas can be stitched together using simple x and y translation. This straightening process may create visible seams and distortion, as we warp the images to account for changing camera angle. Once the panoramas are straightened and aligned, we perform another seam finding and blending step to produce the final super panorama. If the camera height (i.e. image scale) or angle changes a lot during the super panorama, there may be additional ghosting at the panorama seams and the final panorama may "drift".

###Issues
While the pipeline presumably accepts stitching directions of either LEFT, RIGHT, UP, and DOWN, it has only been tested on LEFT and thus probably only works for LEFT.
