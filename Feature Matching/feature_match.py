import cv2
import numpy as np
import matplotlib.pyplot as plt

path = '..\\Resources\\reeses_puffs.png'
reeses = cv2.imread(path,0)
path = '..\\Resources\\many_cereals.jpg'
cereals = cv2.imread(path,0)

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(reeses,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(cereals,cmap='gray')
plt.suptitle('Original Images')

#Brute-force detection with ORB descriptors
orb = cv2.ORB_create()
key_points1,descriptors1 = orb.detectAndCompute(reeses,None)
key_points2,descriptors2 = orb.detectAndCompute(cereals,None)

#Brute-force matching with ORB descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = bf.match(descriptors1,descriptors2)

#Sorting matches based on each match distance attribute
matches = sorted(matches, key=lambda x:x.distance)

#Drawing lines of features that match
orb_matches = cv2.drawMatches(reeses,key_points1,cereals,key_points2,matches[:25],None,flags=2)

plt.figure(2)
plt.imshow(orb_matches,cmap='gray')
plt.title('ORB Descriptor Matching')

#Brute-force detection with SIFT descriptors
sift = cv2.SIFT_create()
key_points1,descriptors1 = sift.detectAndCompute(reeses,None)
key_points2,descriptors2 = sift.detectAndCompute(cereals,None)

#Brute-force KNN matching with SIFT descriptors
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1,descriptors2,k=2)

#Applying ratio test
good_matches = []

for match1,match2 in matches:
    #Less distance means better match
    if match1.distance < 0.75*match2.distance:
        good_matches.append([match1])

#Drawing lines of features that match
sift_matches = cv2.drawMatchesKnn(reeses,key_points1,cereals,key_points2,good_matches,None,flags=2)

plt.figure(3)
plt.imshow(sift_matches,cmap='gray')
plt.title('SIFT Descriptor Matching')

#FLANN Based Matching
sift = cv2.SIFT_create()
key_points1,descriptors1 = sift.detectAndCompute(reeses,None)
key_points2,descriptors2 = sift.detectAndCompute(cereals,None)

#Getting FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

#Applying FLANN parameters
flann = cv2.FlannBasedMatcher(index_params,search_params)
#KNN matching with FLANN
matches = flann.knnMatch(descriptors1,descriptors2,k=2)

#Applying ratio test
good_matches = []

for match1,match2 in matches:
    #Less distance means better match
    if match1.distance < 0.75*match2.distance:
        good_matches.append([match1])

#Drawing lines of features that match
flann_matches = cv2.drawMatchesKnn(reeses,key_points1,cereals,key_points2,good_matches,None,flags=0)

plt.figure(4)
plt.imshow(flann_matches,cmap='gray')
plt.title('FLANN Matching')

#FLANN Based Matching
sift = cv2.SIFT_create()
key_points1,descriptors1 = sift.detectAndCompute(reeses,None)
key_points2,descriptors2 = sift.detectAndCompute(cereals,None)

#Getting FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params = dict(checks=50)

#Applying FLANN parameters
flann = cv2.FlannBasedMatcher(index_params,search_params)
#KNN matching with FLANN
matches = flann.knnMatch(descriptors1,descriptors2,k=2)

#Creating a mask
matches_mask = [[0,0] for i in range(len(matches))]

#Applying ratio test
for i,(match1,match2) in enumerate(matches):
    #Less distance means better match
    if match1.distance < 0.75*match2.distance:
        matches_mask[i] = [1,0]

#Drawing parameters
draw_params = dict(matchColor=(0,255,0),
                    singlePointColor=(255,0,0),
                    matchesMask = matches_mask,
                    flags=0)

#Drawing lines of features that match
flann_matches = cv2.drawMatchesKnn(reeses,key_points1,cereals,key_points2,matches,None,**draw_params)

plt.figure(5)
plt.imshow(flann_matches,cmap='gray')
plt.title('FLANN Matching')

plt.show()