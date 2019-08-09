import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Downsampling if needed
def downsample_image(image, reduce_factor):
    for i in range(0, reduce_factor):
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image

# Image Reader object
class ImageReader(object):
    def __init__(self, path, image, use_mask=True, flip_left=True, downsampling=0):
        self.path = path
        self.mirror_coords = self.__detect_border(image)
        self.use_mask = use_mask
        self.flip_left = flip_left
        self.downsampling = downsampling
        self.files = os.listdir(path)
        self.current_it = 0
        self.nb_images = len(self.files)
    
    #Detecting the bottom mirror coords
    def __detect_border(self, image):
        areas = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        print(hsv)
        
        verde_bajos = np.array([35,50,50])
        verde_altos = np.array([107, 255, 255])
        
        mask = cv2.inRange(hsv, verde_bajos, verde_altos)
        
        plt.imshow(mask)
        
        points_line = []
        
        major_version = cv2.__version__.split('.')[0]

        if major_version == '3':
            _, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i in range(len(cnts)):
            fitted_area = cv2.contourArea(cnts[i])
            print(i, fitted_area)
            areas.append((fitted_area,cnts[i]))

        sorted_areas = sorted(areas, reverse=True, key=lambda x: x[0])
            
        for i in [0,1]:
           ellipse = cv2.fitEllipse(sorted_areas[i][1])
           points_line.append((ellipse[0][0],ellipse[0][1]))
           cv2.ellipse(image, ellipse, (0, 0, 255), 10)
                    
           print("Center of ellipse is ({}, {}), area {}".format(ellipse[0][0], ellipse[0][1], fitted_area))
        
        return points_line
    
    # Reading an image
    def read_image(self):
        filename = self.files[self.current_it]
        original_img = cv2.imread(os.path.join(self.path, filename), 0)
        if self.downsampling > 0:
            original_img = downsample_image(original_img, self.downsampling)
        
        if self.use_mask:
            X, Y = np.meshgrid(np.arange(original_img.shape[1]), np.arange(original_img.shape[0]))
            right_mask = np.zeros(original_img.shape[:2], dtype='uint8')
            xA, yA = self.mirror_coords[0]
            xB, yB = self.mirror_coords[1]
            if self.downsampling > 0:
                xA /= 2 ** self.downsampling
                yA /= 2 ** self.downsampling
            right_mask[(Y - yA - (yB - yA) / (xB - xA) * (X - xA)) >= 0] = 255

            right_img = cv2.bitwise_and(original_img, original_img, mask=right_mask)
            left_img = cv2.bitwise_and(original_img, original_img, mask=(255 - right_mask))
        else:
            right_img = left_img = original_img

        if self.flip_left:
            left_img = np.flip(left_img, axis=1)

        self.current_it = (self.current_it + 1) % len(self.files)
        return original_img, left_img, right_img, filename

    def reset(self):
        self.current_it = 0


