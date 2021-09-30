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

#A) Imagem Original
plt.subplot(151), plt.imshow(img, "gray"), plt.title("Imagem Original")

original = np.fft.fft2(img)
center = np.fft.fftshift(original)

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

############################## Ideal Filter - Model ##################################
ideal1 = idealFilterLP(1.0,img.shape)
plt.subplot(151), plt.imshow(np.abs(ideal1), "gray"), plt.title("Ideal LP 1.0")

ideal15 = idealFilterLP(1.5,img.shape)
plt.subplot(152), plt.imshow(np.abs(ideal15), "gray"), plt.title("Ideal LP 1.5")

ideal50 = idealFilterLP(50,img.shape)
plt.subplot(153), plt.imshow(np.abs(ideal50), "gray"), plt.title("Ideal LP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

############################## Ideal Filter - Aplied ##################################

LowPassCenter = center * idealFilterLP(1.0,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(151), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Ideal LP 1.0")

LowPassCenter = center * idealFilterLP(1.5,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(152), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Ideal LP 1.5")

LowPassCenter = center * idealFilterLP(50,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(153), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Ideal LP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

################################## Butterworth Filter - Model ##########################
butterworth1 = butterworthLP(1.0,img.shape,20)
plt.subplot(151), plt.imshow(np.abs(butterworth1), "gray"), plt.title("Butterworth LP 1.0")

butterworth15 = butterworthLP(1.5,img.shape,20)
plt.subplot(152), plt.imshow(np.abs(butterworth15), "gray"), plt.title("Butterworth LP 1.5")

butterworth50 = butterworthLP(50,img.shape,20)
plt.subplot(153), plt.imshow(np.abs(butterworth50), "gray"), plt.title("Butterworth LP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

################################## Butterworth Filter - Aplied ##########################

LowPassCenter = center * butterworthLP(1.0,img.shape,10)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(151), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Butterworth LP 1.0")

LowPassCenter = center * butterworthLP(1.5,img.shape,10)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(152), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Butterworth LP 1.5")

LowPassCenter = center * butterworthLP(50,img.shape,10)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(153), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Butterworth LP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

################################## Gaussian Filter - Model ##########################

gaussian1 = gaussianLP(1.0,img.shape)
plt.subplot(151), plt.imshow(np.abs(gaussian1), "gray"), plt.title("Gaussian LP 1.0")

gaussian15 = gaussianLP(1.5,img.shape)
plt.subplot(152), plt.imshow(np.abs(gaussian15), "gray"), plt.title("Gaussian LP 1.5")

gaussian50 = gaussianLP(50,img.shape)
plt.subplot(153), plt.imshow(np.abs(gaussian50), "gray"), plt.title("Gaussian LP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)

################################## Gaussian Filter - Aplied ##########################

LowPassCenter = center * gaussianLP(1.0,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(151), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian LP 1.0")

LowPassCenter = center * gaussianLP(1.5,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(152), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian LP 1.5")

LowPassCenter = center * gaussianLP(50,img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)
plt.subplot(153), plt.imshow(np.abs(inverse_LowPass), "gray"), plt.title("Gaussian LP 50")

plt.show()
plt.figure(figsize=(6.4 * 5, 4.8 * 5), constrained_layout=False)






