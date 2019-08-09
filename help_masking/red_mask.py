# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
imagen = cv2.imread('images/Example/3_Red_markers.png')
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

#Rango para color rojo
rojo_bajos1 = np.array([0,65,75], dtype=np.uint8)
rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
rojo_bajos2 = np.array([240,65,75], dtype=np.uint8)
rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)

#Máscaras con inRange()
mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)

#Juntar ambas máscaras de rojo
mask = cv2.add(mascara_rojo1, mascara_rojo2)
print("One")

#Mostramos la mascara final y la imagen
cv2.imshow('Filtered red', mask)
print("Two")
cv2.imshow('Initial', imagen)
print("Three")
cv2.imwrite('images/Example/3_Mask_red.png', mask)

#Salir con ESC
while(1):
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
 
cv2.destroyAllWindows()
print("Caramba")



