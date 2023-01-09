import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '..\\Resources\\Nadia_Murad.jpg'
nadia = cv2.imread(path,0)
path = '..\\Resources\\Denis_Mukwege.jpg'
denis = cv2.imread(path,0)
path = '..\\Resources\\solvay_conference.jpg'
solvay = cv2.imread(path,0)

plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(nadia,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(denis,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(solvay,cmap='gray')
plt.tight_layout()
plt.suptitle('Original Images')

#Loading Haar cascade classifier for face detection
path = '..\\Resources\\haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(path)

def detect_face(img):
    """
    Detects faces in an image and draws a rectangle at faces
    """
    face_img = img.copy()

    face_rectangles = face_cascade.detectMultiScale(face_img)

    for (x,y,w,h) in face_rectangles:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    
    return face_img

#Getting detected faces from image
result_nadia = detect_face(nadia)
result_denis = detect_face(denis)
result_solvay = detect_face(solvay)

plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(result_nadia,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(result_denis,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(result_solvay,cmap='gray')
plt.tight_layout()
plt.suptitle('Face Detection')

#Adjusting face detection to include scale factor and minimum neighbors
def adjusted_detect_face(img):
    """
    Detects faces in an image and draws a rectangle at faces
    """
    face_img = img.copy()

    face_rectangles = face_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)

    for (x,y,w,h) in face_rectangles:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    
    return face_img

result_solvay = adjusted_detect_face(solvay)
plt.figure(3)
plt.imshow(result_solvay,cmap='gray')
plt.title('Face Detection with Scale Factor and Minimum Neighbors')

#Loading Haar cascade classifier for eye detection
path = '..\\Resources\\haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(path)

def detect_eyes(img):
    face_img = img.copy()

    eye_rectangles = eye_cascade.detectMultiScale(face_img,scaleFactor=1.2,minNeighbors=5)

    for (x,y,w,h) in eye_rectangles:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,255,255),10)
    
    return face_img

#Getting detected eyes from image
result_nadia = detect_eyes(nadia)
result_denis = detect_eyes(denis)
result_solvay = detect_eyes(solvay)

plt.figure(4)
plt.subplot(1,3,1)
plt.imshow(result_nadia,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(result_denis,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(result_solvay,cmap='gray')
plt.tight_layout()
plt.suptitle('Eye Detection')


#Testing face detection from video capture
capture = cv2.VideoCapture(0)

while True:
    ret,frame = capture.read()
    frame = adjusted_detect_face(frame)

    cv2.imshow('Video Face Detection', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
plt.show()