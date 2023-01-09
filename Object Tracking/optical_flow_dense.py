import cv2
import numpy as np
import matplotlib.pyplot as plt

#Starting video capture
capture = cv2.VideoCapture(0)

#Reading frames
ret, frame1 = capture.read()

#Grab initial image frame
prev_img = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

#Apply a hsv mask 
hsv_mask = np.zeros_like(frame1)
hsv_mask[:,:,1] = 255

while True:
    #Read current frame
    ret,frame2 = capture.read()

    #Get current frame
    next_img = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    #Getting optical flow using default values

    #Returns vector flow cartesian information
    flow = cv2.calcOpticalFlowFarneback(prev_img,next_img,None,0.5,3,15,3,5,1.2,0)

    #We want to convert it into polar coordinated with magnitude and angle for hue and saturation of HSV
    magnitude,angle = cv2.cartToPolar(flow[:,:,0],flow[:,:,1],angleInDegrees=True)

    #Look at half of the hues
    hsv_mask[:,:,0] = angle/2

    #Normalize magnitude for HSV value channel
    hsv_mask[:,:,2] = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)

    #Conver HSV to BGR
    bgr = cv2.cvtColor(hsv_mask,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame',bgr)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break
    prev_img = next_img

capture.release()
cv2.destroyAllWindows()

