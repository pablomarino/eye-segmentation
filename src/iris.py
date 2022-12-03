# -*- coding: utf-8 -*-
import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

def extraer_iris(imagen,centro,radio_pupila):
    img = preprocesado(imagen,centro,radio_pupila)
    return extraer_iris_umbral_polar(img, centro, radio_pupila)
    #return extraer_iris_sobel_polar(img, centro, radio_pupila)

def preprocesado(imagen,centro,radio_pupila):
    img = np.copy(imagen)

    # Radio de la imagen, la mitad del ancho de la pantalla
    radio_imagen = int(img.shape[1]/2)
    
    # Convierto de polar a cartesiano
    linear_iris = cv.linearPolar(img,centro,radio_imagen,cv.INTER_CUBIC)
    

    #linear_iris[linear_iris<50]=255
    
    # Binarizo con OTSU
    _, img_linear_iris = cv.threshold(linear_iris, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)



    # Opening para eliminar pestañas
    '''
    morph_kernel = np.ones((2,9))#cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_linear_iris = cv.morphologyEx(img_linear_iris, cv.MORPH_OPEN, morph_kernel, iterations=2)
    '''
    morph_kernel = np.ones((9,5))#cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_linear_iris = cv.morphologyEx(img_linear_iris, cv.MORPH_OPEN, morph_kernel, iterations=2)
    
    '''
    #t = cv.adaptiveThreshold(linear_iris, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 115, 1)
    plt.imshow(t, cmap='gray',  interpolation='nearest')
    plt.show()
    '''
    '''
    plt.imshow(img_linear_iris, cmap='gray',  interpolation='nearest')
    plt.show()
    '''
    
    # Elimino la parte de la pupila y lo que quede mas alla de 4 veces radio de pupila
    img_linear_iris = img_linear_iris[:,radio_pupila*2:radio_pupila*8]
    
    
    return img_linear_iris

def extraer_iris_umbral_polar(imagen,center,radio_pupila):
    img = np.copy(imagen)

    # busco limite del iris, para cada linea de la imagen busco el cambio de blanco a negro
    limites = []
    
    # Almaceno el primer pixel blanco de cada fila
    for i in range(len(img)):
        for j in range(len(img[0])):
            if (img[i][j]>0):
                limites.append(j)
                break
    
    # hallo la media
    limite = round(np.average(limites))
    
    # Elimino valores lejanos a la media
    tmp_limite = 0
    iteraciones = -1
    while(limite!=tmp_limite):
        tmp_limite=limite
        max_d=np.amax(limites)
        min_d=np.amin(limites)
        # dispersion acrual de valores lo divido por 4 para que queden fuera los
        # valores que estan mas alla de la mitad del rango por la derecha o izquierda
        dispersion = (max_d-min_d)//4
        for i in range(len(limites)-1,0,-1):
            if(abs(limite-limites[i])>dispersion):
                #limites[i] = 0
                del limites[i]
        
        '''
        plt.plot(limites)
        plt.show()
        '''
        limite = int(round(np.average(limites)))
        
        iteraciones += 1
        #print ("Iris it",iteraciones,"radio ",tmp_limite,">",limite," ",dispersion,"=",max_d,"-",min_d,(100*len(limites)/320),"% de datos utilizados")
        
        '''
        cv.line(img, (limite,0), (limite,240), (100,100,100), 2)        
        plt.imshow(img, cmap='gray',  interpolation='nearest')
        plt.show()  
        '''
        
    ## divido por 2 para corregir tamaño de imagen polar
    return int(limite/2)

def extraer_iris_sobel_polar(imagen,center,radio_pupila):
    return 10


