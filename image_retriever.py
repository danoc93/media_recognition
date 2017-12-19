import utils
import numpy as np
from vocabulary_tree import Vocabulary_Tree_Searcher
from sift_extractor import SIFT_Extractor

###
# Class to retrieve best matches using a trained Vocabulary Tree.
# cfg can be read from 'config.json'.
# This class assumes the same configuration for the trained model.
###

class Image_Retriever:
	
	def __init__(self, cfg):
		# SIFT: limit the number of keypoints to read from an image.
		self.siftExt = SIFT_Extractor(cfg["maxNumberOfKeyPoints"])

		# Configuration.
		self.savedModelDataPath = cfg["projectPath"] + cfg["savedModelDataPath"]
		self.maxNumOfMatches = cfg["maxNumOfMatches"]
		self.numDataPoints = cfg["maxNumberOfDataPoints"]

		# Internal structures.
		self.tree = None
		self.prepared = False
		self.search = None
		self.clusterEntropies = None


	'''
	Prepares the image retriever by loading a saved model.
	'''
	def prepare(self):
		
		print('** Preparing Image Retriever **')
		self.tree = utils.loadModel(self.savedModelDataPath)
		
		if self.tree is not None:
			print('Model loaded from', self.savedModelDataPath)
		else:
			print('! Could not load model.')
			return

		# Search object for looking up matches.
		self.search = Vocabulary_Tree_Searcher(
			self.tree.meanDescriptorClusters, self.tree.indexedTree)

		# Entropies for the leaves.
		self.clusterEntropies = self.tree.leafClusterEntropies

		# Max number of matches limit.
		self.maxNumOfMatches = np.minimum(self.maxNumOfMatches, 
			len(self.tree.sourceToLeafIndex))

		print("MATCHER READY -> maxNumOfMatches: %d" % self.maxNumOfMatches)

		self.prepared = True


	'''
	Returns the closest maxNumOfMatches image labels to the image in targetPath.
	The file order is preserved.
	'''
	def findBestMatchingLabels(self, targetPath):

		best_files = self.findBestMatchingImages(targetPath)
		for i in range( len(best_files) ):
			best_files[i] = utils.getLabelFromFileName(best_files[i])

		return best_files


	'''
	Gets at most maxNumOfMatches matches from the trained data for an image.
	'''
	def findBestMatchingImages(self, targetPath):

		if not self.prepared:
			print('Retriever is not prepared! Call prepare() first.')
			return

		# Get the descriptors and define its bag of words (leaf clusters).
		targetDescriptors = self.siftExt.extractFeatures(targetPath)
		histogram, targets = self.getHistogram(targetDescriptors)

		# Database of trained images!
		#Forward index.
		imageToClusterMap = self.tree.sourceToLeafIndex
		#Inverted index.
		clusterToImageMap = self.tree.leavesToImageIndex

		# Algorithm: 
		#1. Get the score for each word in the query image.
		#2. Get the score for the same word in a database image.
		#3. Compute their absolute difference.
		#4. Repeat for all words, and then proceed to the next image.
		scores = []
		for db_img in targets:
			
			score = 0
			for clusterId in histogram:

				#Trained image matches for the current cluster.
				clusterToImage = clusterToImageMap[clusterId]

				if db_img not in clusterToImage:
					continue

				q_i = histogram[clusterId]
				d_i = clusterToImage[db_img]

				# L1 norm score weighted by entropy of the level!
				score += self.clusterEntropies[clusterId] * \
				( np.abs(q_i - d_i) - np.abs(q_i) - np.abs(d_i) )

			scores.append(score + 2)

		# Minimizing!
		scores = np.array(scores)
		best_n = scores.argsort()[:self.maxNumOfMatches]

		return [targets[i] for i in best_n]


	'''
	Given a list of descriptors, compute the weighted histogram of the leaves.
	'''
	def getHistogram(self, targetDescriptors):

		# Get the best leaf cluster for each descriptor and count the frequency.
		leafOccurencesInTarget = {}
		# We only care about these!
		targets = set()
		for descriptor in targetDescriptors:
			bestClusterId = self.search.getBestLeafIdFromCluster(descriptor, 0)
			
			for f in self.tree.leavesToImageIndex[bestClusterId]:
				targets.add(f)

			if bestClusterId not in leafOccurencesInTarget:
				leafOccurencesInTarget[bestClusterId] = 0
			leafOccurencesInTarget[bestClusterId] += 1

		return leafOccurencesInTarget, list(targets)


