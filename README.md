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

### Stitching images

You will need to at minimum update the `config.yaml` to point to the correct directory where your images are stored. The images should contain a sequential number within the filename which corresponds to the timestamp or order in which the images should be stitched. 