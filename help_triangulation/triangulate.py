import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import pickle
import config
from random import sample


# how many images to triangulate and plot
how_many = 2

[stereocalib_retval, M1, d1, M2, d2, R, T, E, F, valid_images, img_left_points, img_right_points] = (
    pickle.load(open(config.PKL_FILE, "rb"))
)
# distortion coeffs seem to be wrong
#d1 = d2 = np.zeros_like(d1)

# Get projection matrices for triangulation
(rectification_l, rectification_r, projection_l, projection_r, disparityToDepthMap, ROI_l, ROI_r) = (
    cv2.stereoRectify(M1, d1, M2, d2, config.IMG_SHAPE[:2], R, T, None, None, None, None, None, 0,
                      alpha=config.ALPHA)
)


# The second argument are the two vertex of the mirror, used for computing right and left masks
imagen = cv2.imread('images/X_Lab_Images/image1.jpg')
image_reader = ImageReader(config.BASE_PATH, imagen, use_mask=True, flip_left=True,
                           downsampling=config.DOWNSAMPLING)
plot_these = sample(valid_images, how_many)

elev = -105
azim = -110

img_counter = 0
plot_counter = 0
fig = plt.figure()
for _ in range(image_reader.nb_images):
    original_img, left_img, right_img, filename = image_reader.read_image()

    if filename in plot_these:
        plot_counter += 1
        left_corners = np.squeeze(img_left_points[img_counter]).T
        right_corners = np.squeeze(img_right_points[img_counter]).T

    # if it is a valid image we must increment img_counter to extract the proper left and right corners from
    # img_left_points and img_right_points
    if filename in valid_images:
        img_counter += 1

    if filename not in plot_these:
        continue

    #Hallar pto 3D con coords de las boxes (sacar centro)
    pts4D = cv2.triangulatePoints(projection_l, projection_r, left_corners, right_corners).T
    # convert from homogeneous coordinates to 3D
    pts3D = pts4D[:, :3]/np.repeat(pts4D[:, 3], 3).reshape(-1, 3)

    # fig = plt.figure(filename)
    plt.subplot(how_many, 2, plot_counter)
    plt.imshow(original_img, 'gray')

    # 0, 1, 2
    plot_counter += 1
    Xs = pts3D[:, 0]
    Ys = pts3D[:, 1]
    Zs = pts3D[:, 2]
    ax = fig.add_subplot(how_many, 2, plot_counter, projection='3d')
    # ax.scatter(0, 0, 0, c='b', marker='o')
    ax.scatter(Xs[1:], Ys[1:], Zs[1:], c='b', marker='o')
    ax.scatter(Xs[0], Ys[0], Zs[0], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)

plt.tight_layout()
plt.show()
