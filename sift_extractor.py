import cv2
import numpy as np


###
# Class to extract image features using SIFT.
###
class SIFT_Extractor:

	def __init__(self, maxNumberOfKeyPoints, scale = False, scaleFactor = 200):
		self.SIFT = cv2.xfeatures2d.SIFT_create(maxNumberOfKeyPoints)
		self.scale = scale
		self.scaleFactor = scaleFactor 



	''' 
	Return the features from a grayScaleImage. 
	'''
	def extractFeaturesFromImage(self, grayImage, includeKeyPoints = False):

		keyPoints, descriptors = self.SIFT.detectAndCompute(grayImage, None)
		return (keyPoints, descriptors) if includeKeyPoints else descriptors


	''' 
	Return the features from the image located at imagePath 
	'''
	def extractFeatures(self, imagePath, includeKeyPoints = False):

		grayScaleImage = self.getGrayscaleImage(imagePath)
		return self.extractFeaturesFromImage(grayScaleImage, includeKeyPoints)


	'''
	Return a map of file:descriptors for all the images in the list.
	'''
	def getFileToDescriptorsMap(self, listOfImagePaths):

		fileToDescriptorsMap = {}
		for imagePath in listOfImagePaths:
			fileToDescriptorsMap[imagePath] = self.extractFeatures(imagePath)

		return fileToDescriptorsMap


	'''
	Return a list of all descriptors in the file:descriptors map.
	'''
	def extractAllDescriptors(self, fileToDescriptorsMap):

		allDescriptors = []
		for imagePath in fileToDescriptorsMap:
			allDescriptors.extend( fileToDescriptorsMap[imagePath] )

		return allDescriptors


	'''
	Visualize the keypoints for an image.
	'''
	def visualizeKeyPoints(self, imagePath):

		grayScaleImage = self.getGrayscaleImage(imagePath)

		keyPoints, descriptors = \
			self.extractFeaturesFromImage(grayScaleImage, True)

		result = cv2.drawKeypoints(grayScaleImage,keyPoints, None, None, \
			flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

		cv2.imshow('dst_rt', result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		cv2.waitKey(1)


	'''
	Get a grayscale version of the image.
	'''
	def getGrayscaleImage(self, imagePath):

		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if self.scale:
			image = cv2.resize(image, (self.scaleFactor, self.scaleFactor)) 
		return image