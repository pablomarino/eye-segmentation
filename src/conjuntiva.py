# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
import math
from math import atan
from matplotlib import pyplot as plt

def extraer_conjuntiva(img,centro,radio_pupila_iris):
    return extraer_conjuntiva_morph(img,centro,radio_pupila_iris)
    #return extraer_conjuntiva_polar_suma(img,centro,radio_pupila_iris)

def define_circle(p1, p2, p3):
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((int(cx), int(cy)), int(radius))

def aproximar_media(v):
    tmp_v = v[v!=0]
    print(len(v),len(tmp_v))

def extraer_conjuntiva_morph(imagen,centro,radio_pupila_iris):
    edges = cv.Canny(imagen,50,200)
    
    # Elimino pupila e iris pero sin afectar al perimetro de conjuntiva
    cv.circle(edges,centro, 3*radio_pupila_iris//4,(1, 1, 1), -1)
    cv.circle(edges,(centro[0]-50,centro[1]), 2*radio_pupila_iris//3,(1, 1, 1), -1)
    cv.circle(edges,(centro[0]+50,centro[1]), 2*radio_pupila_iris//3,(1, 1, 1), -1)
    #     
    morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 3))
    
    edges = cv.dilate(edges, morph_kernel, iterations=2)
    edges =cv.medianBlur(edges,11,11)
    edges = cv.erode(edges, morph_kernel, iterations=2)
    
    arc_size = 50
    
    # Derecha
    pts_right=[0]*arc_size
    radius = len(imagen[0])-centro[0]
    for i in range(arc_size):
        for j in range(radius):
            angle = (math.pi*(i-arc_size//2))/180
            x = centro[0] +round( j * math.cos(angle))
            y = centro[1] +round( j * math.sin(angle))
            if (edges[y][x]==255):
                pts_right[i] = j
                break
    index = np.array(pts_right).argmax()
    angle = (math.pi*(index-arc_size//2))/180
    radius = pts_right[index]
    x = centro[0] +round( radius * math.cos(angle))
    y = centro[1] +round( radius * math.sin(angle))
    pt_right=(x,y)
    
    # arriba
    pts_top=[0]*arc_size
    radius = len(imagen)-centro[1]
    for i in range(arc_size):
        for j in range(radius):
            angle = (math.pi*(i+270-arc_size//2))/180
            x = centro[0] +round( j * math.cos(angle))
            y = centro[1] +round( j * math.sin(angle))
            if (edges[y][x]==255):
                pts_top[i] = j
                break
    index = np.array(pts_top).argmax()
    angle = (math.pi*(index+270-arc_size//2))/180
    radius = pts_top[index]
    x = centro[0] +round( radius * math.cos(angle))
    y = centro[1] +round( radius * math.sin(angle))
    pt_top=(x,y)
    
    # Izquierda
    pts_left=[0]*arc_size
    radius = len(imagen[0])-centro[0]
    for i in range(arc_size):
        for j in range(radius):
            angle = (math.pi*(i+180-arc_size//2))/180
            x = centro[0] +round( j * math.cos(angle))
            y = centro[1] +round( j * math.sin(angle))
            if (edges[y][x]==255):
                pts_left[i] = j
                break
    index = np.array(pts_left).argmax()
    angle = (math.pi*(index+180-arc_size//2))/180
    radius = pts_left[index]
    x = centro[0] +round( radius * math.cos(angle))
    y = centro[1] +round( radius * math.sin(angle))
    pt_left=(x,y)
    
    # abajo
    pts_bottom=[0]*arc_size
    radius = len(imagen)-centro[1]
    for i in range(arc_size):
        for j in range(radius):
            angle = (math.pi*(i+90-arc_size//2))/180
            x = centro[0] +round( j * math.cos(angle))
            y = centro[1] +round( j * math.sin(angle))
            if (edges[y][x]==255):
                pts_bottom[i] = j
                break
    index = np.array(pts_bottom).argmax()
    angle = (math.pi*(index+90-arc_size//2))/180
    radius = pts_bottom[index]
    x = centro[0] +round( radius * math.cos(angle))
    y = centro[1] +round( radius * math.sin(angle))
    pt_bottom=(x,y)
    
    
    #Ver puntos sobre canny
    cv.circle(edges,pt_top, 5,(128, 1, 1), -1)
    cv.circle(edges,pt_left, 5,(128, 1, 1), -1)
    cv.circle(edges,pt_right, 5,(128, 1, 1), -1)
    cv.circle(edges,pt_bottom, 5,(128, 1, 1), -1)
    
    plt.imshow(edges, cmap='gray', interpolation='nearest')
    plt.show()
    '''
    '''
    
    center_top, radius_top = define_circle(pt_left, pt_top, pt_right)
    center_bottom, radius_bottom = define_circle(pt_left, pt_bottom, pt_right)
    

    return pt_left,pt_right,pt_top,pt_bottom,center_top, radius_top,center_bottom, radius_bottom


'''
    Crea un kernel gaussiano
'''
def gkern(l=5, sig=1.):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)      

'''
    Da un valor medio de intensidad de los pixeles de cada fila
'''
def sum_rows(linear_iris):
    row_values = []
    
    for i in range(len(linear_iris)):
        valid_pixels = 0
        tmp = 0
        for j in range(len(linear_iris[i])-1,0,-1):
            # evito los pixeles negros que aparecen a la derecha por conversion a polar
            if(valid_pixels != 0 or linear_iris[i][j]!=0):
                # evito valores mas extremos
                if(linear_iris[i][j]>80 and linear_iris[i][j]<253):
                    valid_pixels+=1
                    tmp+=linear_iris[i][j]
        row_values.append(tmp/valid_pixels)

    return row_values
    
def get_max_min(row_sobel):
    # Listas que contienen todos los maximos y minimos
    tmp_max = []
    tmp_min = []
    # Lista que contiene los 2 maximos y 2 minimos que considero mas significativos
    final_max = [0,0]
    
    # Obtengo minimos y maximos
    for i in range(1,len(row_sobel)):
        # paso de positivo a negativo, es un maximo
        if(row_sobel[i-1]>0 and row_sobel[i]<0):
            tmp_max.append(i)
        # paso de negativo a positivo, es un minimo
        elif(row_sobel[i-1]<0 and row_sobel[i]>0):
            tmp_min.append(i)
    
    # me aseguro de tener solo dos maximos uno en la parte superior de 
    # la imagen polar y otro en la inferior
    max_top = []
    max_bottom = []
    for i in range(len(tmp_max)):
        if tmp_max[i]<len(row_sobel)//2:
            max_top.append(tmp_max[i])
        else:
            max_bottom.append(tmp_max[i])
    # si no encuentro maximos me los invento
    if(len(max_top)==0):
        max_top.append(len(row_sobel)//4)
    if(len(max_bottom)==0):
        max_bottom.append(3*(len(row_sobel)//4))
        
    final_max = [np.array(max_top).mean(),np.array(max_bottom).mean()]
    
    # obtengo el minimo que esta entre los dos maximos
    mins = []
    for i in range(len(tmp_min)):
        if (tmp_min[i]>final_max[0]) and (tmp_min[i]<=final_max[1]):
            mins.append(tmp_min[i])
    
    if(len(mins)==0):
        mins.append(final_max[0]+(final_max[1]-final_max[0])//2)
    
    final_min = np.array(mins).mean()
    
    plt.scatter(final_max[0],0, s=100)
    plt.scatter(final_max[1],0, s=100)
    
    plt.scatter(final_min,0, s=20)

    
    plt.plot(row_sobel)
    plt.show()
    
    #convierto los valores polares a un angulo
    final_max[0] =-90+final_max[0]*1.5;
    final_max[1] =-90+final_max[1]*1.5;

    final_min = -90+final_min*1.5;
    
    return final_max[0],final_max[1],final_min

def get_boundaries(imagen,centro,radio_pupila_iris,ang_left,ang_right,ang_top):
    
    edges = cv.Canny(imagen,50,200)
    
    # Elimino pupila e iris
    cv.circle(edges,centro, radio_pupila_iris,(1, 1, 1), -1)
    #     
    morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    edges = cv.dilate(edges, morph_kernel, iterations=2)
    edges =cv.medianBlur(edges,21,21)
    edges = cv.erode(edges, morph_kernel, iterations=2)
    
    #.morphologyEx(edges, cv.MORPH_OPEN, morph_kernel, iterations=2)
    
    plt.imshow(edges, cmap='gray', interpolation='nearest')
    plt.show()
    
    pt_left = (centro[0]-radio_pupila_iris,centro[1])
    pt_right = (centro[0]+radio_pupila_iris,centro[1])
    pt_top = (centro[0],centro[1]-radio_pupila_iris)
    pt_bottom = (centro[0],centro[1]+radio_pupila_iris)
    return pt_left,pt_right,pt_top,pt_bottom

def extraer_conjuntiva_polar_suma(imagen,centro,radio_pupila_iris):
    img = np.copy(imagen)

    # Radio de la imagen, la mitad del ancho de la pantalla
    radio_imagen = int(img.shape[1]/2)
    
    # Convierto de polar a cartesiano
    linear_iris = cv.linearPolar(img,centro,radio_imagen,cv.INTER_CUBIC+cv.WARP_FILL_OUTLIERS)
    
    # Elimino la parte de la pupila e iris
    linear_iris = linear_iris[:,radio_pupila_iris*2:]
    
    #sumo valor de intensidad de las filas
    row_values = sum_rows(linear_iris)
    #plt.plot(row_values)
    # aplico un suavizado gaussiano al resultado
    ksize = 40
    sigma = 150
    vect = gkern(ksize,sigma)[ksize//2]
    row_values = np.convolve(row_values, vect, 'same') #'valid') 
    plt.plot(row_values)
    # Busco maximos, minimos con 1º derivada de los resultados usando un sobel
    # Donde primera derivada  pasa de + a - (MAX) o - a + (min) hay corte con eje 
    row_sobel = cv.Sobel(row_values,cv.CV_64F,0,1,ksize=5)
    
    # con los maximos y minimos obtengo el angulo de 3 puntos 
    ang_left,ang_right,ang_top = get_max_min(row_sobel)
    
    # canny de la imagen y con estos angulos desde la pupila busco donde 
    # hay colision con el perimetro de la conjuntiva
    
    
    return get_boundaries(imagen,centro,radio_pupila_iris,ang_left,ang_right,ang_top)
