import cv2
import matplotlib.pyplot as plt
import imutils
import numpy as np
from numpy import asarray
from PIL import Image
from scipy import ndimage

#A)
widthImage, heightImage = 512, 512
filePath = 'img/quadro.png'
fileName = (filePath, "PNG")

ObjPillow = Image.new("RGB", (widthImage, heightImage))
setPillow = ObjPillow.load()

for row in range(heightImage):
    for col in range(widthImage):
        color = (000)
        rev_col, rev_row = widthImage - col - 1, heightImage - row - 1
        setPillow[col, row] = color
        setPillow[rev_col, row] = color
        setPillow[col, rev_row] = color
        setPillow[rev_col, rev_row] = color

for row in range(206, 306):
    for col in range(206, 306):
        color = (255, 255, 255)
        setPillow[col, row] = color

ObjPillow.save(*fileName)
rgbToGrey = Image.open(filePath).convert('L')
rgbToGrey.save(filePath)
rgbToGrey.show()

img = cv2.imread('img/quadro.png', 0)
plt.subplot(151), plt.imshow(img, "gray"), plt.title("Imagem Original")


#B)
amplitude = np.fft.fft2(img)
plt.subplot(152), plt.imshow(np.log(1+np.abs(amplitude)), "gray"), plt.title("Espectro Amplitude")


#C)
fase = np.angle(amplitude)
plt.subplot(153), plt.imshow(fase, "gray"), plt.title("Espectro Fase")


#D)
center = np.fft.fftshift(amplitude)
plt.subplot(154), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Espectro Central")

plt.show()

#E)
imagem = cv2.imread('img/quadro.png')
altura, largura = imagem.shape[:2]
cv2.waitKey(0)

#rotacao 40
ponto = (largura / 2, altura / 2) #ponto no centro da figura
rotacao = cv2.getRotationMatrix2D(ponto, 40, 1.0)
rotacionado = cv2.warpAffine(imagem, rotacao, (largura, altura))
cv2.imshow("Rotacionado 40 graus", rotacionado)
cv2.waitKey(0)

cv2.imwrite('img/quadro40.png', rotacionado)

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

#img rotacionada
img_rotacionada40 = cv2.imread('img/quadro40.png', 0)
plt.subplot(151), plt.imshow(img_rotacionada40, "gray"), plt.title("Rotação 40º")


#amplitude
amplitude = np.fft.fft2(img_rotacionada40)
plt.subplot(152), plt.imshow(np.log(1+np.abs(amplitude)), "gray"), plt.title("Espectro Amplitude")


#fase
fase = np.angle(amplitude)
plt.subplot(153), plt.imshow(fase, "gray"), plt.title("Espectro Fase")


#central
center = np.fft.fftshift(amplitude)
plt.subplot(154), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Espectro Central")

plt.show()

#F)
#translação
deslocamento = np.float32([[1, 0, -30], [0, 1, -70]]) #parametros para movimentar a imagem 
deslocado = cv2.warpAffine(imagem, deslocamento, (largura, altura))
cv2.imshow("Cima e esquerda", deslocado)
cv2.waitKey(0)

cv2.imwrite('img/quadro_trans.png', deslocado)

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

#img transladada
img_trans = cv2.imread('img/quadro_trans.png', 0)
plt.subplot(151), plt.imshow(img_trans, "gray"), plt.title("Imagem Transladada")

#amplitude
amplitude = np.fft.fft2(img_trans)
plt.subplot(152), plt.imshow(np.log(1+np.abs(amplitude)), "gray"), plt.title("Espectro Amplitude")

#fase
fase = np.angle(amplitude)
plt.subplot(153), plt.imshow(fase, "gray"), plt.title("Espectro Fase")

#central
center = np.fft.fftshift(amplitude)
plt.subplot(154), plt.imshow(np.log(1+np.abs(center)), "gray"), plt.title("Espectro Central")

plt.show()