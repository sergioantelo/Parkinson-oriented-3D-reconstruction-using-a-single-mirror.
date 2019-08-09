import cv2
import os

PATTERN_SIZE = (9, 6)

# Own base path
BASE_PATH = "images/X_Lab_Images"

MIRROR_COORDS = [] 

DOWNSAMPLING = 0

PKL_FILE = "calibration_whitey.pkl"

ALPHA = -1

IMG_SHAPE = cv2.imread(os.path.join(BASE_PATH, os.listdir(BASE_PATH, )[0])).shape[:2]
