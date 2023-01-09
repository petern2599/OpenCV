import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '..\\Resources\\car_plate.jpg'
car_plate = cv2.imread(path)

path = '..\\Resources\\rusland19.jpg'
another_car_plate = cv2.imread(path)

path = '..\\Resources\\Lada_Vesta_(cropped).jpg'
another_car_plate2 = cv2.imread(path)

path = '..\\Resources\\5ebbc81b85600a286f4de97e.jpg'
another_car_plate3 = cv2.imread(path)

#Load Haar cascade classifier for Russian license plate detection
path = '..\\Resources\\haarcascade_russian_plate_number.xml'
russian_plate_cascade = cv2.CascadeClassifier(path)

def detect_and_blur_plate(img):
    """
    Detects license plate and blurs it
    """
    #Get two copies of the image for detection and blurring, respectively
    #This is to avoid blurring the rectangle drawn
    plate_img = img.copy()
    blur_img = img.copy()

    #Get position and dimensions of detected license plate
    plate_rectangle = russian_plate_cascade.detectMultiScale(plate_img,scaleFactor=1.2,minNeighbors=5)

    for (x,y,w,h) in plate_rectangle:
        #Draw rectangle around detected license plate
        cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,0,255),15)
        #Blurs ROI of license plate
        plate_img[y:y+h,x:x+w] = cv2.blur(blur_img[y:y+h,x:x+w],ksize=(30,30))

    return plate_img

#Get results
result = detect_and_blur_plate(car_plate)
result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
car_plate = cv2.cvtColor(car_plate,cv2.COLOR_BGR2RGB)

plt.figure(1)
plt.imshow(car_plate)
plt.title('Original Image')

plt.figure(2)
plt.imshow(result)
plt.title('Plate Detection and Blurring')

result = detect_and_blur_plate(another_car_plate)
result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
another_car_plate = cv2.cvtColor(another_car_plate,cv2.COLOR_BGR2RGB)

plt.figure(3)
plt.imshow(another_car_plate)
plt.title('Original Image')

plt.figure(4)
plt.imshow(result)
plt.title('Plate Detection and Blurring')

result = detect_and_blur_plate(another_car_plate2)
result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
another_car_plate2 = cv2.cvtColor(another_car_plate2,cv2.COLOR_BGR2RGB)

plt.figure(5)
plt.imshow(another_car_plate2)
plt.title('Original Image')

plt.figure(6)
plt.imshow(result)
plt.title('Plate Detection and Blurring')

result = detect_and_blur_plate(another_car_plate3)
result = cv2.cvtColor(result,cv2.COLOR_BGR2RGB)
another_car_plate3 = cv2.cvtColor(another_car_plate3,cv2.COLOR_BGR2RGB)

plt.figure(7)
plt.imshow(another_car_plate3)
plt.title('Original Image')

plt.figure(8)
plt.imshow(result)
plt.title('Plate Detection and Blurring')

plt.show()
