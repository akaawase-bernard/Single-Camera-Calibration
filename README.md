# Single Camera Calibration

This project performs camera calibration using a set of images of a checkerboard pattern. The calibration process calculates the camera matrix and distortion coefficients, which are essential for various computer vision applications. 

## Features
- **Checkerboard Pattern Detection**: Automatically detects and refines the positions of checkerboard corners in a series of images.
- **Camera Calibration**: Computes the camera matrix and distortion coefficients.
- **Image Rotation**: Rotates input images 90 degrees clockwise for proper alignment.
- **Visualization**: Saves and plots images with detected checkerboard corners, including a progress bar for image processing.

## Prerequisites
- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- PyYAML
- tqdm

## Installation
First, clone the repository:
```sh
git clone https://github.com/akaawase-bernard/single_camera_calibration.git
cd single_camera_calibration
```

## Usage
```python calibrate_single_camera.py```

## The Structure
```
workdir/
├── detected_corners_00.png
├── detected_corners_01.png
├── ...
├── camera_matrix.npy
├── dist_coeffs.npy
```
## Contributing
Feel free to submit issues and enhancement requests.

## License
This project is licensed under the MIT License.
