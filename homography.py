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
		Compute the number of inliers in the tomography between source and destination.
		It also takes a best match so it can be drawn along the others.
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


	def plotTogeter(self, sourceImage, destinationImage, bestMatch, matchesMask, good, transform = None):

		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
				   singlePointColor = None,
				   matchesMask = matchesMask, # draw only inliers
				   flags = 2)

		img1 = cv2.imread(sourceImage, 0)
		img2 = cv2.imread(destinationImage, 0)
		imgbest = cv2.imread(bestMatch, 0)

		kp1, des1 = self.siftExt.extractFeatures(sourceImage, True)
		kp2, des2 = self.siftExt.extractFeatures(destinationImage, True)

		img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

		f, axarr = plt.subplots(2,2)
		axarr[0,1].imshow(img3, 'gray')
		axarr[0,1].axis('off')
		axarr[0,1].set_title('Homography')

		axarr[1,0].imshow(imgbest, 'gray')
		axarr[1,0].axis('off')
		axarr[1,0].set_title('Top From Tree Only')
		
		axarr[1,1].imshow(img2, 'gray')
		axarr[1,1].axis('off')
		axarr[1,1].set_title('Top From Tree + Less Inliners')
		
		axarr[0,0].imshow(img1, 'gray')
		axarr[0,0].set_title('Target To Detect')
		axarr[0,0].axis('off')

		plt.show()

