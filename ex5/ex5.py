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

#A) Imagem Original
plt.subplot(151), plt.imshow(img, "gray"), plt.title("Original Image")

original = np.fft.fft2(img)
center = np.fft.fftshift(original)

plt.show()

############################## Ideal Filter - Model ##################################
ideal1 = idealFilterHP(1.0,img.shape)
plt.subplot(151), plt.imshow(np.abs(ideal1), "gray"), plt.title("Ideal HP 1.0")

ideal15 = idealFilterHP(1.5,img.shape)
plt.subplot(152), plt.imshow(np.abs(ideal15), "gray"), plt.title("Ideal HP 1.5")

ideal50 = idealFilterHP(50,img.shape)
plt.subplot(153), plt.imshow(np.abs(ideal50), "gray"), plt.title("Ideal HP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

############################## Ideal Filter - Aplied ##################################

LowPassCenter = center * idealFilterHP(1.0,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(151), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Ideal HP 1.0")

LowPassCenter = center * idealFilterHP(1.5,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(152), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Ideal HP 1.5")

LowPassCenter = center * idealFilterHP(50,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(153), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Ideal HP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

################################## Butterworth Filter - Model ##########################
butterworth1 = butterworthHP(1.0,img.shape,20)
plt.subplot(151), plt.imshow(np.abs(butterworth1), "gray"), plt.title("Butterworth HP 1.0")

butterworth15 = butterworthHP(1.5,img.shape,20)
plt.subplot(152), plt.imshow(np.abs(butterworth15), "gray"), plt.title("Butterworth HP 1.5")

butterworth50 = butterworthHP(50,img.shape,20)
plt.subplot(153), plt.imshow(np.abs(butterworth50), "gray"), plt.title("Butterworth HP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

################################## Butterworth Filter - Aplied ##########################

LowPassCenter = center * butterworthHP(1.0,img.shape,10)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(151), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Butterworth HP 1.0")

LowPassCenter = center * butterworthHP(1.5,img.shape,10)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(152), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Butterworth HP 1.5")

LowPassCenter = center * butterworthHP(50,img.shape,10)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(153), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Butterworth HP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

################################## Gaussian Filter - Model ##########################

gaussian1 = gaussianHP(1.0,img.shape)
plt.subplot(151), plt.imshow(np.abs(gaussian1), "gray"), plt.title("Gaussian HP 1.0")

gaussian15 = gaussianHP(1.5,img.shape)
plt.subplot(152), plt.imshow(np.abs(gaussian15), "gray"), plt.title("Gaussian HP 1.5")

gaussian50 = gaussianHP(50,img.shape)
plt.subplot(153), plt.imshow(np.abs(gaussian50), "gray"), plt.title("Gaussian HP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

################################## Gaussian Filter - Aplied ##########################

LowPassCenter = center * gaussianHP(1.0,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(151), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian HP 1.0")

LowPassCenter = center * gaussianHP(1.5,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(152), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian HP 1.5")

LowPassCenter = center * gaussianHP(50,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(153), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian HP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)