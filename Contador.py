import cv2
import numpy as np
#Numpy esta por defecto en python

varGauss=9
varKrnel=4

#Importamos imagen original
img=cv2.imread('/home/zeus/Escritorio/Python/Proyecto1/euros.jpg')

#Aplicamos filtro de color gris
Gimg=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)


#Tomamos la imagen gris para el suavisado
suave=(cv2.GaussianBlur(Gimg,(varGauss,varGauss),0))


#Terminar con canny el ruido
Canny=cv2.Canny(suave,60,100)

#KERNEL ES LA MATRIZ TRANSFORMADA A ENTEROS 
kernel=np.ones((varKrnel,varKrnel),np.uint8)

#TIPO DE MORPHOLOGIA PARA RUIDO
cierre=cv2.morphologyEx(Canny,cv2.MORPH_CLOSE,kernel)

#Encontramos los contornos externos 
contorno,jerarquia=cv2.findContours(cierre.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#Mandamos a consola la cantidad encontrada de contornos
print("Monedas encontradas: ",format(len(contorno)))


#Dibujamos con la imagen oroginal, cantidad de contornos, color del contorno y grosor 
cv2.drawContours(img,contorno,-1,(251,60,50),2)


#Imprime imagenes de los procesos 
cv2.imshow("Gray",Gimg)
cv2.imshow("Gauss",suave)
cv2.imshow("Canny",Canny)
cv2.imshow("Contornos",img)
cv2.waitKey(0)
cv2.destroyAllWindows()