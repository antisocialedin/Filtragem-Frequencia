import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, exp

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

img = cv2.imread("img/lena.jpg", 0)

#A)
plt.subplot(151), plt.imshow(img, "gray"), plt.title("Imagem Original")

original = np.fft.fft2(img)
center = np.fft.fftshift(original)

plt.show()

#B)
LowPass = idealFilterLP(50,img.shape)
plt.subplot(152), plt.imshow(np.abs(LowPass), "gray"), plt.title("Ideal LP")

LowPass = butterworthLP(50,img.shape,20)
plt.subplot(153), plt.imshow(np.abs(LowPass), "gray"), plt.title("Butterworth LP")

LowPass = gaussianLP(50,img.shape)
plt.subplot(154), plt.imshow(np.abs(LowPass), "gray"), plt.title("Gaussian LP")

plt.show()

plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

#C) 

LowPassCenter = center * idealFilterLP(50,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(161), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Ideal LP")

LowPassCenter = center * butterworthLP(50,img.shape,10)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(162), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Butterworth LP")

LowPassCenter = center * gaussianLP(50,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(163), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian LP")

plt.show()