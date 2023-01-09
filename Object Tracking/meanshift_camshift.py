import cv2
import numpy as np
import matplotlib.pyplot as plt

#Starting video capture
capture = cv2.VideoCapture(0)

#Reading frames
ret,frame = capture.read()

#Load Haar cascade classifier for face detection
path = 'C:\\Users\\peter\\opencv\\Resources\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path)

#Detect face
face_rectangles = face_cascade.detectMultiScale(frame)

#Get region of detected face which will be the tracking window
(face_x,face_y,width,height) = tuple(face_rectangles[0])
track_window = (face_x,face_y,width,height)

#Define ROI of the detected face in the frame
roi = frame[face_y:face_y+height,face_x:face_x+width]

#Convert ROI from BGR to HSV
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

#Calculate the color histogram of the HSV ROI
roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])

#Normalize the histogram
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

#Define termination criteria
termination_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret,frame = capture.read()

    if ret == True:
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        #Calculate the back propagation to see how the pixels of the image fit the histogram model of the ROI
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        #MeanShift (uncomment to use)
        # ret,track_window = cv2.meanShift(dst,track_window,termination_criteria)
        # x,y,w,h = track_window
        # img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)

        #CAMShift to resize roi (uncomment to use)
        ret,track_window = cv2.CamShift(dst,track_window,termination_criteria)
        points = cv2.boxPoints(ret)
        points = np.int0(points)
        img2 = cv2.polylines(frame,[points],True,(255,0,0),5)

        cv2.imshow('img',img2)

        k=cv2.waitKey(1) & 0xFF

        if k==27:
            break
    else:
        break

cv2.destroyAllWindows()
capture.release()