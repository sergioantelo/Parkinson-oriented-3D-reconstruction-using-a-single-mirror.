# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
#Introduce your image
imagen = cv2.imread('images/Example/2_Blue_markers.png')
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

#Red color range
azul_bajos = np.array([100,65,75], dtype=np.uint8)
azul_altos = np.array([130, 255, 255], dtype=np.uint8)

#Masking
mask = cv2.inRange(hsv, azul_bajos, azul_altos)

#Showing the final mask and the original image
cv2.imshow('Filtered blue', mask)
cv2.imshow('Initial', imagen)

#Save the image in your address
cv2.imwrite('images/Example/2_Mask_blue.png', mask)

#Exit with esc key
while(1):
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
 
cv2.destroyAllWindows()
