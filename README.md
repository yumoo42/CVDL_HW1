# CVDL_HW1
This is a PyQt5-based graphical application that integrates various computer vision and image processing functionalities. The main features include:

## Camera Calibration:
- Detects chessboard corners in images to compute camera intrinsic parameters (e.g., focal length, principal point) and extrinsic parameters (e.g., rotation matrix, translation vector).
- Corrects image distortion and displays the results.

## Augmented Reality (AR):
- Projects letters onto a virtual 3D plane based on camera parameters in a chessboard background, simulating augmented reality effects.

## Stereo Disparity Map:
- Computes disparity maps from left and right images for depth estimation.

## Feature Detection and Matching (SIFT):
- Detects feature points in a single image and visualizes them.
- Matches feature points between two images and visualizes the matching results.
