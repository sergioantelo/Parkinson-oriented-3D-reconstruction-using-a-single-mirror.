import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import pickle
import config

# The second argument are the two vertex of the mirror, used for computing right and left masks
imagen = cv2.imread('images/Z_Camera/DSCN8348.jpg')
image_reader = ImageReader(config.BASE_PATH, imagen, use_mask=True, flip_left=True,
                           downsampling=config.DOWNSAMPLING)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cbrow, cbcol = (config.PATTERN_SIZE)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow*cbcol,3), np.float32)
objp[:,:2] = np.mgrid[0:cbrow,0:cbcol].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for i in range(image_reader.nb_images):
    print("Processing {} / {}".format(i + 1, image_reader.nb_images))

    original_img, left_img, right_img, filename = image_reader.read_image()

    #find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    #find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    gray = right_img
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, config.PATTERN_SIZE, None)
    
    #left_found, left_corners = cv2.findChessboardCorners(left_img, config.PATTERN_SIZE, flags=find_chessboard_flags)
    #right_found, right_corners = cv2.findChessboardCorners(right_img, config.PATTERN_SIZE, flags=find_chessboard_flags)

    if ret == True:
        print("True")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(gray, config.PATTERN_SIZE, corners2, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        
#cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)

pickle.dump([ret, mtx,
             dist, rvecs, tvecs], open(config.PKL_FILE_SIMPLE, "wb"))

# Compare calibration parameters with those that we would obtain using simple calibration
print("Final calibration for camera:")
print(mtx)
print(dist)
print("===================")

