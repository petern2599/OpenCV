import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '..\\Resources\\sammy_face.jpg'
img = cv2.imread(path)

plt.figure(1)
plt.imshow(img)
plt.title('Original Image')

edges = cv2.Canny(image=img, threshold1=127,threshold2=127)

plt.figure(2)
plt.imshow(edges)
plt.title('Canny Edge Detection (Tight Thresholds)')

edges = cv2.Canny(image=img, threshold1=0,threshold2=255)

plt.figure(3)
plt.imshow(edges)
plt.title('Canny Edge Detection (Wide Thresholds)')

#Calculating the median value
med_val = np.median(img)

#Calculating the lower and upper threshold values
lower_thresh = int(max(0,0.7*med_val))
upper_thresh = int(min(255,1.3*med_val))

edges = cv2.Canny(image=img, threshold1=lower_thresh,threshold2=upper_thresh)

plt.figure(4)
plt.imshow(edges)
plt.title('Canny Edge Detection (Calculated Thresholds)')

#Blurring image to remove noise to detect edges better with threshold values calculated
blurred_img = cv2.blur(img,ksize=(5,5))
edges = cv2.Canny(image=blurred_img, threshold1=lower_thresh,threshold2=upper_thresh)

plt.figure(5)
plt.imshow(edges)
plt.title('Canny Edge Detection on Blurred Image (Calculated Thresholds)')


plt.show()