# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

imagen = cv2.imread('images/Whitey/Set_2/IMG_1945.jpg')
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

#Rango para color rojo
verde_bajos = np.array([49,50,50])
verde_altos = np.array([107, 255, 255])

#Máscaras con inRange()
mask = cv2.inRange(hsv, verde_bajos, verde_altos)
print("One")

#Mostramos la mascara final y la imagen
plt.imshow(mask)
plt.imshow(imagen)
cv2.imshow('Filtered green', mask)
print("Two")
cv2.imshow('Initial', imagen)
print("Three")
cv2.imwrite('images/Example/1_Mask_green_IMG.png', mask)



#Salir con ESC
while(1):
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
 
cv2.destroyAllWindows()
print("Caramba")
