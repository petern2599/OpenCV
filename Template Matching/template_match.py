import cv2
import numpy as np
import matplotlib.pyplot as plt

#Getting the full image
path = "..\\Resources\\sammy.jpg"
full_img = cv2.imread(path)
full_img = cv2.cvtColor(full_img,cv2.COLOR_BGR2RGB)

plt.figure(1)
plt.imshow(full_img)

#Getting the template
path = '..\\Resources\\sammy_face.jpg'
face_img = cv2.imread(path)
face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)

plt.figure(2)
plt.imshow(face_img)

#All the 6 methods for comparison in a list
#Note how we are using strings, later on we'll use the eval() function to convert to function
#The first 4 methods look a maximum whereas the last 2 look for a minimum for a match
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

figure_index=3
for m in methods:
    #Creating a copy of the image
    full_img_copy = full_img.copy()

    #Using eval() function to get function
    method = eval(m)

    #Tempplate matching
    res = cv2.matchTemplate(full_img_copy,face_img,method)

    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

    #Since these methods look for a minimum for a match unlike the others
    #need to make sure that location is correct
    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        rect_top_left = min_loc
    else:
        rect_top_left = max_loc
    
    height,width,channels = face_img.shape

    rect_bottom_right = (rect_top_left[0]+width,rect_top_left[1]+height)

    cv2.rectangle(full_img_copy,rect_top_left,rect_bottom_right,(255,0,0),10)

    plt.figure(figure_index)
    plt.subplot(1,2,1)
    plt.imshow(res)
    plt.title('Heatmap of Template Matching')

    plt.subplot(1,2,2)
    plt.imshow(full_img_copy)
    plt.title('Detection of Template')
    plt.suptitle(m)
    
    figure_index+=1

plt.show()