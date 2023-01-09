import cv2
import numpy as np
import matplotlib.pyplot as plt

#Setting up params for Shi-Tomasi corner detection
corner_track_params = dict(maxCorners=10,qualityLevel=0.3,minDistance=7,blockSize=7)

#Setting up params for Lucas-Kanade optical flow function

#Smaller window size is more sensitive to noise due to large motion
#Larger window size is less sentive to large motions but not sensitive enough for smaller motion

#Setting max level to 2 for image pyramid which affects the image resolution

#Provided two criteria to exchange speed of tracking vs accuracy of tracking:
#   Count -> 10 iterations (larger count = more exhaustive)
#   EPS -> 0.03 (lower epsilon = faster)
lk_params = dict(winSize=(200,200),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))

#Getting video capture
capture = cv2.VideoCapture(0)

#Read initial frame
ret,prev_frame = capture.read()

#Convert frame to grayscale
prev_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

#Points to track
prev_points = cv2.goodFeaturesToTrack(prev_gray,mask=None,**corner_track_params)

#Create a mask
mask = np.zeros_like(prev_frame)

while True:
    #Read current frame
    ret,frame = capture.read()

    #Convert current frame to grayscale
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Calculating optical flow
    next_points,status,error = cv2.calcOpticalFlowPyrLK(prev_gray,frame_gray,prev_points,None,**lk_params)

    #Status vector of 1 indicates optical flow was found
    good_new = next_points[status==1]
    good_prev = prev_points[status==1]

    for i, (new,prev) in enumerate(zip(good_new,good_prev)):
        #Get the position of the new frame
        x_new,y_new = new.ravel()
        #Get the position of the previous frame
        x_prev,y_prev = prev.ravel()

        #Create a line on the mask from the previous position to the new position
        mask = cv2.line(mask,(int(x_new),int(y_new)),(int(x_prev),int(y_prev)),(0,255,0),3)

        #Draw a circle on the frame to keep track of positioning of significant features
        frame = cv2.circle(frame,(int(x_new),int(y_new)),8,(0,0,255),-1)
    
    #Combine the mask and the current frame
    img = cv2.add(frame,mask)
    cv2.imshow('tracking',img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

    #Copy the current frame to the previous frame
    prev_gray = frame_gray.copy()
    #Reshape the feature points
    prev_points = good_new.reshape(-1,1,2)

capture.release()
cv2.destroyAllWindows()