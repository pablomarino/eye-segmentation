import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt

def extraer_pupila(imagen):
    img = preprocesado(imagen)
    return extraer_pupila_umbral_iter(img)
    #return extraer_pupila_umbral_polar(img)


def preprocesado(img):
    # Imagen difuminada para eliminar el ruido
    img_blur = cv.GaussianBlur(img, (5, 5), 0)
    # img_bluCCr = cv.medianBlur(img,15,15)

    # Creo una mascara para la region central de la imagen para aislar la pupila
    mascara_radio = 70
    mascara_pupila = np.zeros(img.shape, np.uint8) * 255
    cv.circle(mascara_pupila, (round(mascara_pupila.shape[1] / 2), round(mascara_pupila.shape[0] / 2)), mascara_radio,
              (1, 1, 1), -1)
    img_mask = np.copy(img_blur)
    for i in range(mascara_pupila.shape[0]):
        for j in range(mascara_pupila.shape[1]):
            if (mascara_pupila[i][j] == 0):
                img_mask[i][j] = 255

    '''
    plt.imshow(img_mask, cmap='gray',  interpolation='nearest')
    plt.show()
    '''
    # Creo una imagen umbralizada en la que solo aparece la pupila
    umbral_pupila = 45

    img_pupila = cv.inRange(img_mask, 0, umbral_pupila)
    ret, img_pupila = cv.threshold(img_pupila, np.average(img_pupila), 1, cv.THRESH_BINARY)
    '''
    plt.imshow(img_pupila, cmap='gray',  interpolation='nearest')
    plt.show()
    '''
    # Opening para eliminar pestañas
    morph_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_pupila_open = cv.morphologyEx(img_pupila, cv.MORPH_OPEN, morph_kernel, iterations=2)
    return img_pupila_open


def extraer_pupila_umbral_polar(img):
    
    # Obtengo el centro de la pupila
    m = cv.moments(img)
    center = (round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"]))
    
    # Convierto a coordenadas polares
    polar_pupila = cv.linearPolar(img,center,img.shape[0]/2,cv.INTER_CUBIC+cv.WARP_FILL_OUTLIERS)
    '''
    plt.imshow(polar_pupila, cmap='gray',  interpolation='nearest')
    plt.show()
    '''
    tmp = 0
    for i in range(len(polar_pupila)):
        tmp += np.argmax(polar_pupila[i]<1)/2
    radio = int(tmp /len(polar_pupila))
    # recorro la imagen de derecha a izquierda y cuando encuentro el blanco de la pupila paro
    # me quedo la media de los valores de todas las filas
    # con ese valor de radio relleno la pupila para recalcular el centro
    
    # me he estado quedando con los valores mas pequeños al irme cojo el mas grande
    #print(len(polar_pupila),radio)
    return center, radio

def extraer_pupila_umbral_iter(img):
    # aproximo el area de la pupila de manera iterativa para eliminar brillos
    tmp_area_pupila = 0
    area_pupila = 1
    iteraciones = -1

    while ((area_pupila - tmp_area_pupila) >= 1):
        tmp_area_pupila = area_pupila
        # Obtengo el centro de la pupila
        m = cv.moments(img)
        center = (round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"]))

        # calculo el radio de la pupila r = sqrt(area / pi)
        area_pupila = int(math.ceil(np.sum(np.ravel(img))))
        radio_pupila = int(math.ceil((area_pupila / math.pi) ** (1 / 2)))

        # superpongo esta aproximacion de la pupila para eliminar brillos
        cv.circle(img, center, radio_pupila, (1, 1, 1), -1)

        iteraciones += 1
        #print("Pupila it", iteraciones, "area ", tmp_area_pupila, ">", area_pupila,"centro", center)

        '''
        plt.imshow(img, cmap='gray',  interpolation='nearest')
        plt.show()
        '''
        
    # El area de la pupila tiene el mismo tamaño que en la iteracion anterior
    return center, radio_pupila