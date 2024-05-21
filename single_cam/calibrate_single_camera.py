import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from tqdm import tqdm

def load_config(file_path):
    """Load the configuration parameters from a YAML file."""
    with open(file_path, 'r') as file:
        print('Has sucessfully loaded the config file')
        return yaml.safe_load(file)

def perform_camera_calibration(config):
    """
    Perform camera calibration using a set of images of a checkerboard pattern.
    
    Parameters:
    config (dict): Configuration parameters including:
                   - image_directory (str): Path pattern to the input images.
                   - checkerboard_dims (tuple): Dimensions of the checkerboard pattern.
                   - square_size (float): Size of a square in real-world units.
                   - output_dir (str): Directory to save output images and results.
                   
    Returns:
    tuple: Camera matrix and distortion coefficients.
    
    Author: Akaawase Bernard
    """
    # Get parameters from config
    image_directory = config['image_directory']
    checkerboard_dims = tuple(config['checkerboard_dims'])
    square_size = config['square_size']
    output_dir = config['output_dir']

    # Get list of image paths
    image_paths = sorted(glob.glob(image_directory))
    image_list = [cv.imread(img_path) for img_path in image_paths]

    # Prepare object points based on checkerboard dimensions
    obj_point_template = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    obj_point_template[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)
    obj_point_template *= square_size

    # Arrays to store object points and image points
    obj_points = []
    img_points = []

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for idx, image in enumerate(tqdm(image_list, desc="Processing images")):
        # Rotate image 90 degrees clockwise
        image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray_image, checkerboard_dims, None)

        if ret:
            # Refine corner positions
            refined_corners = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), 
                                              (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(refined_corners)
            obj_points.append(obj_point_template)

            # Draw and save the detected corners
            cv.drawChessboardCorners(image, checkerboard_dims, refined_corners, ret)
            output_path = os.path.join(output_dir, f'detected_corners_{idx}.png')
            #cv.imwrite(output_path, image)

            # Plot the image with corners
            plt.figure(figsize=(15, 15))  # Larger plot size
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            plt.title(f'Sciencestical 4: {idx:03}', fontsize=15, weight='bold')  # Bold title
            plt.axis('off')
            plot_path = os.path.join(output_dir, f'plot_{idx:03}.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    # Perform camera calibration
    img_shape = gray_image.shape[::-1]
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, img_shape, None, None)
    
    # Display results
    print('Reprojection error:', retval)
    print('Camera matrix:\n', camera_matrix)
    print('Distortion coefficients:', dist_coeffs)
    
    # Save calibration results
    np.save(os.path.join(output_dir, 'camera_matrix.npy'), camera_matrix)
    np.save(os.path.join(output_dir, 'dist_coeffs.npy'), dist_coeffs)

    return camera_matrix, dist_coeffs


# Load configuration and calibrate camera
config = load_config('config.yaml')
cam_matrix, cam_dist = perform_camera_calibration(config)
