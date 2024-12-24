from setuptools import setup, find_packages

setup(
    name='panorama_maker',
    version='0.1.0',
    description='A tool for stitching together images from agricultural scenes using LightGlue and other tools',
    author='GEMINI Breeding',
    author_email='ikuyehara@ucdavis.edu',
    url='https://github.com/GEMINI-Breeding/Panorama-Maker.git',  # Replace with your repo
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'torch',
        'numpy',
        'PyYAML',
        'torchvision',
        'ipykernel',
        'matplotlib',
        'stitching'
        # Add other dependencies
    ],
    dependency_links=[
        'https://github.com/cvg/LightGlue.git#egg=LightGlue'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
