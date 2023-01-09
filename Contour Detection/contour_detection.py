import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '..\\Resources\\internal_external.png'
img = cv2.imread(path,0)

plt.figure(1)
plt.imshow(img,cmap='gray')
plt.title('Original Image')

#Finding contours
contours,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

#Setting up variables
external_contours = np.zeros(img.shape)

for i in range(len(contours)):
    #In hierarchy, external contours are indicated by -1
    if hierarchy[0][i][3] == -1:
        #Drawing countours in external contours variable
        cv2.drawContours(external_contours,contours,i,255,-1)

plt.figure(2)
plt.imshow(external_contours,cmap='gray')
plt.title('External Contours')

internal_contours = np.zeros(img.shape)

for i in range(len(contours)):
    #In hierarchy, internal contours are any value except for -1
    #Values that are the same are contours that are grouped together
    if hierarchy[0][i][3] != -1:
        #Drawing countours in internal contours variable
        cv2.drawContours(internal_contours,contours,i,255,-1)

plt.figure(3)
plt.imshow(internal_contours,cmap='gray')
plt.title('Internal Contours')

internal_contours = np.zeros(img.shape)

for i in range(len(contours)):
    #In hierarchy, internal contours are any value except for -1
    #Values that are the same are contours that are grouped together
    if hierarchy[0][i][3] == 4:
        #Drawing countours in internal contours variable
        cv2.drawContours(internal_contours,contours,i,255,-1)

plt.figure(4)
plt.imshow(internal_contours,cmap='gray')
plt.title('Internal Contours')
plt.show()