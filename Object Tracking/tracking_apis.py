import cv2
import numpy as np
import matplotlib.pyplot as plt

def ask_for_tracker():
    """
    Ask for user input on the type of tracking API to use
    """
    print("Welcome! What Tracker API would you like to use?")
    print('Enter 0 for BOOSTING ')
    print("Enter 1 for MIL ")
    print("Enter 2 for KCF ")
    print("Enter 3 for TLD ")
    print("Enter 4 for MEDIANFLOW ")

    choice = input("Please select your tracker: ")

    if choice == '0':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif choice == '1':
        tracker = cv2.TrackerMIL_create()
    elif choice == '2':
        tracker = cv2.TrackerKCF_create()
    elif choice == '3':
        tracker = cv2.legacy.TrackerTLD_create()
    elif choice == '4':
        tracker = cv2.legacy.TrackerMedianFlow_create()

    return tracker

#Get tracker API
tracker = ask_for_tracker()

#Create string of tracker API name
tracker_name = str(tracker).split()[1]

#Start video capture
capture = cv2.VideoCapture(0)

#Read initial frame
ret,frame = capture.read()

#Allow user to select ROI of the object to track
roi = cv2.selectROI(frame,False)

#Initialize tracker with the current frame and the ROI of the object
ret = tracker.init(frame,roi)

while True:
    #Read the current frame
    ret,frame = capture.read()
    
    #Get updated ROI of object being tracked
    success, roi = tracker.update(frame)

    #Store ROI dimension and location in tuple
    (x,y,w,h) = tuple(map(int,roi))

    #If tracking is successful, draw a rectangle around object
    if success:
        point1 = (x,y)
        point2 = (x+w,y+h)
        cv2.rectangle(frame,point1,point2,(0,255,0),3)
    #Else say tracking is lost
    else:
        cv2.putText(frame,"Failure to Detect Tracking!", (100,200), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    
    #Put text on frame of the tracker API
    cv2.putText(frame,tracker_name, (20,400), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    cv2.imshow(tracker_name,frame)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()