import cv2
import numpy as np
import matplotlib.pyplot as plt

#Introduce your image 
imagen = cv2.imread('images/Example/3_Red_markers.png')
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

#Red color range
rojo_bajos1 = np.array([0,65,75], dtype=np.uint8)
rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
rojo_bajos2 = np.array([240,65,75], dtype=np.uint8)
rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)

#Double masking
mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)

#Join both red masks
mask = cv2.add(mascara_rojo1, mascara_rojo2)

#Showing the final mask and the original image
cv2.imshow('Filtered red', mask)
cv2.imshow('Initial', imagen)
cv2.imwrite('images/Example/3_Mask_red.png', mask)

#Exit with esc key
while(1):
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
 
cv2.destroyAllWindows()



