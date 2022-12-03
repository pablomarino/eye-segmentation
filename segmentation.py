# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from src.pupila import extraer_pupila
from src.iris import extraer_iris
from src.conjuntiva import extraer_conjuntiva

imagenes = ["aeval1","aevar1","bryanl1","bryanr1",
            "chongpkl2","chongpkr2","kelvinl1","kelvinr1",
            "mazwanl4","mazwanr4","pcll1","pclr1",
            "vimalal1","vimalar1","yannl1","yannr1"]


def overlay(imagen,centro,radio_pupila,radio_iris,pt_left,pt_right,pt_top,pt_bottom,center_t,radius_t,center_b,radius_b):
    img = cv.cvtColor(imagen, cv.COLOR_GRAY2BGR)#np.copy(imagen)
    # Pupila
    cv.circle(img, centro, radio_pupila, (255,0,0),1)
    # Iris
    cv.circle(img, centro, radio_pupila+radio_iris, (0,255,0),1)
    # Conjuntiva
    cv.circle(img, center_t, radius_t, (255,100,255),2)
    cv.circle(img, center_b, radius_b, (255,100,255),2)

    return img
       
def procesar_imagen(imagen):
    img = cv.imread(imagen, 0)
    # Obtengo centro de la pupila y radio
    centro, radio_pupila = extraer_pupila(img) 
    # Obtengo el radio del iris
    radio_iris = extraer_iris(img, centro, radio_pupila)
    # obtengo el area de la conjuntiva
    pt_left,pt_right,pt_top,pt_bottom,center_t,radius_t,center_b,radius_b = extraer_conjuntiva(img, centro, radio_pupila+radio_iris)
    
    output = overlay(img,centro,radio_pupila,radio_iris,pt_left,pt_right,pt_top,pt_bottom,center_t,radius_t,center_b,radius_b)
    plt.imshow(output, cmap='gray', interpolation='nearest')
    plt.savefig("../result/"+imagen+".png")
    plt.show()
    

for i in range(len(imagenes)):
    procesar_imagen("assets/original/"+imagenes[i]+".bmp")
    