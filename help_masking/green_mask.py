import cv2
import numpy as np
import matplotlib.pyplot as plt

#Introduce your image
imagen = cv2.imread('images/Whitey/Set_2/IMG_1945.jpg')
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

#Green color range
verde_bajos = np.array([49,50,50])
verde_altos = np.array([107, 255, 255])

#Masking
mask = cv2.inRange(hsv, verde_bajos, verde_altos)

#Showing the final mask and the original image
cv2.imshow('Filtered green', mask)
cv2.imshow('Initial', imagen)

#Saving the mask to your address
cv2.imwrite('images/Example/1_Mask_green_IMG.png', mask)

#Exit with esc key
while(1):
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
 
cv2.destroyAllWindows()
