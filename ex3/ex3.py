import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

img = cv2.imread("img/lena.jpg", 0)

#A)
plt.subplot(151), plt.imshow(img, "gray"), plt.title("Original Image")

original = np.fft.fft2(img)
center = np.fft.fftshift(original)

plt.show()

#B)
filtro_ideal = idealFilterHP(50,img.shape)
plt.subplot(152), plt.imshow(np.abs(filtro_ideal), "gray"), plt.title("Filtro Ideal HP")

butterworth = butterworthHP(50,img.shape,20)
plt.subplot(153), plt.imshow(np.abs(butterworth), "gray"), plt.title("Butterworth HP")

gaussian = gaussianHP(50,img.shape)
plt.subplot(154), plt.imshow(np.abs(gaussian), "gray"), plt.title("Gaussian HP")

plt.show()

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

#C
central_ideal = center * idealFilterHP(50,img.shape)
filtro_ideal = np.fft.ifftshift(central_ideal)
processada_ideal = np.fft.ifft2(filtro_ideal)
plt.subplot(161), plt.imshow(np.abs(processada_ideal), "gray"), plt.title("Ideal HP")

central_butterworth = center * butterworthHP(50,img.shape,10)
butterworth = np.fft.ifftshift(central_butterworth)
processada_butterworth = np.fft.ifft2(butterworth)
plt.subplot(162), plt.imshow(np.abs(processada_butterworth), "gray"), plt.title("Butterworth HP")

central_gaussiano = center * gaussianHP(50,img.shape)
gaussian = np.fft.ifftshift(central_gaussiano)
processada_gaussiano = np.fft.ifft2(gaussian)
plt.subplot(163), plt.imshow(np.abs(processada_gaussiano), "gray"), plt.title("Gaussian HP")

plt.show()
