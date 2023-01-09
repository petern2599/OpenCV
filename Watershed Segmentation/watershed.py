import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '..\\Resources\\pennies.jpg'
sep_coins = cv2.imread(path)

plt.figure(1)
plt.imshow(sep_coins)
plt.title('Original Image')

#--------Using learned techniques to try to segment coins--------
#Median Blur
blurred_coins = cv2.medianBlur(sep_coins,25)
plt.figure(2)
plt.imshow(blurred_coins)
plt.title('Blurred Image')

#Grayscale
gray_coins = cv2.cvtColor(blurred_coins,cv2.COLOR_BGR2GRAY)
plt.figure(3)
plt.imshow(gray_coins,cmap='gray')
plt.title('Grayscale Image')

#Binary Threshold
ret,thresh = cv2.threshold(gray_coins,thresh=160,maxval=255,type=cv2.THRESH_BINARY_INV)
plt.figure(4)
plt.imshow(thresh,cmap='gray')
plt.title('Inverse Binary Threshold of Image')

#Find Contours
contours,hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins,contours,i,(255,0,0),10)

plt.figure(5)
plt.imshow(sep_coins)
plt.title('External Contour on Image')

#--------Using Watershed Algorithm--------
path = '..\\Resources\\pennies.jpg'
sep_coins = cv2.imread(path)

#Median Blur (using large kernel size for watershed algorithm)
blurred_coins = cv2.medianBlur(sep_coins,35)
plt.figure(6)
plt.imshow(blurred_coins)
plt.title('Blurred Image with Larger Kernel Size')

#Grayscale
gray_coins = cv2.cvtColor(blurred_coins,cv2.COLOR_BGR2GRAY)
plt.figure(7)
plt.imshow(gray_coins,cmap='gray')
plt.title('Grayscale Image')

#Binary Threshold and Otsu method (set thresh to 0 and add otsu to type)
ret,thresh = cv2.threshold(gray_coins,thresh=0,maxval=255,type=cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.figure(8)
plt.imshow(thresh,cmap='gray')
plt.title('Inverse Binary Threshold with Otsu Method of Image')

#Noise Removal - Using Open morphological operator (OPTIONAL)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel=kernel,iterations=2)

plt.figure(9)
plt.imshow(opening,cmap='gray')


#Using Dilation to expand threshold as much as possible to determine background
bg_thresh = cv2.dilate(opening,kernel,iterations=3)
plt.figure(10)
plt.imshow(opening,cmap='gray')
plt.title('Dilation')

#Use Distance Transform to take advantage of binary image to determine foreground points/seeds
dist_tranform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.figure(11)
plt.imshow(dist_tranform,cmap='gray')
plt.title('Distance Transform')

#Apply threshold to foreground points
ret,fg_thresh = cv2.threshold(dist_tranform,thresh=0.7*dist_tranform.max(),maxval=255,type=0)
plt.figure(12)
plt.imshow(fg_thresh,cmap='gray')
plt.title('Foreground Objects')

#Using the Watershed Algorithm to determine the regions where the second threshold did not show 
#compared to the first threshold

#Convert foreground threshold to integers
fg_thresh = np.uint8(fg_thresh)

#Getting the unknown region that is not shown in the foreground threshold compared to the first
#threshold
unknown_region = cv2.subtract(bg_thresh,fg_thresh)
plt.figure(13)
plt.imshow(unknown_region,cmap='gray')
plt.title('Unknown Regions')

#Creating markers using the foreground threshold to essentially create individual labels/groupings
#for each foreground point
ret,markers = cv2.connectedComponents(fg_thresh)
plt.figure(14)
plt.imshow(markers,cmap='gray')
plt.title('Foreground Markers')

#Setting values for markers to differentiate foreground,background,and unknown region
#marker value > 1 -> forground points
#marker value == 1 -> background
#marker value == 0 -> unknown region
markers = markers + 1
markers[unknown_region==255] = 0

plt.figure(15)
plt.imshow(markers,cmap='gray')
plt.title('Markers with Unknown Region')

#Apply markers to Watershed Algorithm
markers = cv2.watershed(sep_coins,markers)
plt.figure(16)
plt.imshow(markers,cmap='gray')
plt.title('Watershed Algorithm')

#Find Contours
contours,hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins,contours,i,(255,0,0),10)

plt.figure(17)
plt.imshow(sep_coins)
plt.title('External Contours on Image')

plt.show()