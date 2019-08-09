# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
imagen = cv2.imread('images/Example/2_Blue_markers.png')
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

#Rango para color rojo
azul_bajos = np.array([100,65,75], dtype=np.uint8)
azul_altos = np.array([130, 255, 255], dtype=np.uint8)

#Máscaras con inRange()
mask = cv2.inRange(hsv, azul_bajos, azul_altos)
print("One")

#Mostramos la mascara final y la imagen
cv2.imshow('Filtered blue', mask)
print("Two")
cv2.imshow('Initial', imagen)
print("Three")
cv2.imwrite('images/Example/2_Mask_blue.png', mask)

#Salir con ESC
while(1):
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
 
cv2.destroyAllWindows()
print("Caramba")