import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import pickle
import config
import math

# Path to the video
videoFile = "vid/X_Lab_Calibrate/calibrate.mp4"

# Reading the video
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = './images/X_Lab_Images/image' +  str(int(x)) + ".jpg";x+=1
        cv2.imwrite(filename, frame)

cap.release()
print ("Done! Ready for the calibration")

# Reading the initial image for the image reader object
imagen = cv2.imread('images/X_Lab_Images/image1.jpg')
image_reader = ImageReader(config.BASE_PATH, imagen, use_mask=True, flip_left=True,
                           downsampling=config.DOWNSAMPLING)
 
img_left_points = []
img_right_points = []
obj_points = []
valid_images = []
first_plot = True
 
# Stereo calibration process
for i in range(image_reader.nb_images):
    print("Processing {} / {}".format(i + 1, image_reader.nb_images))

    original_img, left_img, right_img, filename = image_reader.read_image()

    find_chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    left_found, left_corners = cv2.findChessboardCorners(left_img, config.PATTERN_SIZE, flags=find_chessboard_flags)
    right_found, right_corners = cv2.findChessboardCorners(right_img, config.PATTERN_SIZE, flags=find_chessboard_flags)

    if left_found:
        cv2.cornerSubPix(left_img, left_corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    if right_found:
        cv2.cornerSubPix(right_img, right_corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

    if left_found and right_found:
        img_left_points.append(left_corners)
        img_right_points.append(right_corners)
        objp = np.zeros((np.prod(config.PATTERN_SIZE), 3), np.float32)
        objp[:, :2] = np.mgrid[0:config.PATTERN_SIZE[0], 0:config.PATTERN_SIZE[1]].T.reshape(-1, 2)
        obj_points.append(objp)
        valid_images.append(filename)
        print("Found chessboard in " + filename)

    if left_found and right_found and first_plot and False:
        first_plot = False

        plt.figure()
        plt.subplot(221)
        plt.imshow(
            cv2.drawChessboardCorners(cv2.cvtColor(left_img, cv2.COLOR_GRAY2RGB), config.PATTERN_SIZE, left_corners,
                                      left_found),
        )
        plt.subplot(222)
        plt.imshow(
            cv2.drawChessboardCorners(cv2.cvtColor(right_img, cv2.COLOR_GRAY2RGB), config.PATTERN_SIZE, right_corners,
                                      right_found),
        )
        plt.subplot(223)
        plt.imshow(original_img, "gray")
        plt.show()
        plt.tight_layout()

img_shape = right_img.shape
print("Calibrating...")

obj_points1 = obj_points.copy()
obj_points2 = obj_points.copy()

# Calibrate both cameras separarely for a good initialization of the points
_, cameraMatrix1, distCoeffs1, _, _ = (
    cv2.calibrateCamera(obj_points1, img_left_points, right_img.shape, None, None, None, None)
)
_, cameraMatrix2, distCoeffs2, _, _ = (
    cv2.calibrateCamera(obj_points2, img_right_points, right_img.shape, None, None, None, None)
)

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
stereocalib_flags = (
    cv2.CALIB_USE_INTRINSIC_GUESS
    # cv2.CALIB_FIX_INTRINSIC
    # cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL |
    # cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
)

[stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F] = (
    cv2.stereoCalibrate(obj_points, img_left_points, img_right_points, cameraMatrix1, distCoeffs1, cameraMatrix2,
                        distCoeffs2,
                        right_img.shape, criteria=stereocalib_criteria, flags=stereocalib_flags)
)

pickle.dump([stereocalib_retval, cameraMatrix1,
             distCoeffs1, cameraMatrix2, distCoeffs2,
             R, T, E, F, valid_images, img_left_points, img_right_points], open(config.PKL_FILE, "wb"))

# Compare calibration parameters with those that we would obtain using simple calibration
print("Final calibration for camera 1:")
print(cameraMatrix1)
print(distCoeffs1)
print("===================")

print("Final calibration for camera 2:")
print(cameraMatrix2)
print(distCoeffs2)
print("===================")

_, cameraMatrix, distCoeffs, _, _ = (
    cv2.calibrateCamera(obj_points, img_left_points, left_img.shape, None, None, None, None)
)
print("Calibration for real camera (calibrateCamera):")
print(cameraMatrix)
print(distCoeffs)
