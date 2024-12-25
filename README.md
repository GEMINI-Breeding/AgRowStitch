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
Run "image_stitching_LG_OpenCV.py" from the command line with the path to your config.yaml as the only argument. The two main modes are either affine stitching (projection: "affine" in config.yaml, also reproduced in "affine_image_stitching_LG_Open_CV.py", which ignores the projection parameter) or spherical stitching (projection: "spherical" in config.yaml). Spherical projection is currently recommended.

### Stitching images
Update the "config.yaml" to customize parameters and then pass the path of the config file when running the script. At a minimum, the image directory and output directories must be updated to run the pipeline. The images should contain a sequential number within the filename which corresponds to the timestamp or order in which the images should be stitched. We assume that subsequent images have overlapping areas, but we attempt to stitch only a subset of the images with the forward_limit correspondig to the number of images past the focal image that will be checked for stitching. As the overlap between subsequent images increases, the forward_limit should increase so that as few images as necessary are used. There is a balance between ensuring strong matching so that false matches are ignored and the pipeline failing to find any matches. The min_inliers is the main parameter for changing the confidence between matches, the other confidence values are set by default in "image_stitching_LG_OpenCV.py", but can be adjusted manually. The affine stitching should produce more reliable results, while spherical stitching will produce better stitches when it works.

### Concept
We use SuperPoint to extract image features and LightGlue to identify feature matches between images. The best images are selected to make a single panorama going from the first to the final image where we assume that the camera motion should conform to some constraints (keypoint_prop, xy_ratio, scale_constraint). These images and there features are then represented as OpenCV objects to take advantage of the camera adjustment, seam finding, exposure compensation, and blending in the OpenCV stitching pipeline.
