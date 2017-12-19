import numpy as np
import cv2
from matplotlib import pyplot as plt
from sift_extractor import SIFT_Extractor

class Homography_Finder:

	def __init__(self, minNumberOfMatches = 1, numKeyPoints = 500, scaleFactor = 200):
		
		#Parameters.
		self.minNumberOfMatches = minNumberOfMatches
		self.siftExt = SIFT_Extractor(numKeyPoints)
		self.scaleFactor = scaleFactor

	'''
	Compute the number of inliers in the homography between source and destination.
	This code was adapted to work on Python 3.6 
	https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature_homography.html

	'''
	def findHomography(self, sourceImage, destinationImage):
		
		MIN_MATCH_COUNT = self.minNumberOfMatches

		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)


		img1 = cv2.imread(sourceImage, 0)
		img2 = cv2.imread(destinationImage, 0)

		img1 = cv2.resize(img1, (self.scaleFactor, self.scaleFactor)) 
		img2 = cv2.resize(img2, (self.scaleFactor, self.scaleFactor)) 

		# Initiate SIFT detector
		sift = cv2.xfeatures2d.SIFT_create()

		# find the keypoints and descriptors with SIFT
		kp1, des1 = sift.detectAndCompute(img1,None)
		kp2, des2 = sift.detectAndCompute(img2,None)

		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1,des2,k=2)


		# store all the good matches as per Lowe's ratio test.
		good = []
		for m,n in matches:
			if m.distance < 0.7*n.distance:
				good.append(m)

		inliers = 0
		if len(good)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			matchesMask = mask.ravel().tolist()
			inliers = sum(matchesMask)
			h,w = img1.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			
		else:
			#print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
			matchesMask = None

		return inliers, matchesMask, good



