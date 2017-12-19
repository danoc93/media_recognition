from sklearn.cluster import KMeans
from sift_extractor import SIFT_Extractor
import numpy as np
import utils


###
# Class to train a Vocabulary Tree using the parameters in configuration.
# cfg can be read from 'config.json'.
###
class Tree_Builder:

	def __init__(self, cfg):
		# SIFT: limit the number of keypoints to read from an image.
		self.siftExt = SIFT_Extractor(cfg["maxNumberOfKeyPoints"])

		# K-MEANS: use branching factor.
		self.branchingFactor = cfg["branchingFactor"]
		self.kmModel = KMeans(n_clusters=self.branchingFactor)

		# File Parameters.
		self.trainingDataPath = cfg["projectPath"] + cfg["trainingDataPath"]
		self.savedModelDataPath = cfg["projectPath"] + cfg["savedModelDataPath"]

		# Training Parameters.
		self.numLevels = cfg["maxNumberOfLevels"]
		self.minClusterElements = 3 * self.branchingFactor
		self.numDataPoints = cfg["maxNumberOfDataPoints"]
		self.descriptors = None

		# Model parameters.
		self.indexedTree = {}
		self.leavesToImageIndex = {}
		self.sourceToLeafIndex = {}
		self.meanDescriptorClusters = {}
		self.leafClusterEntropies = {}

		# Utils.
		self.lastIndexedCluster = -1

		# Flags.
		self.trainingComplete = False



	'''
	Recursively construct the hierarchy at the current level.
	Split the data into sub-levels until max-depth reached.
	'''
	def updateIndexedTree(self, level, identifiers, atCluster):

		self.indexedTree[atCluster] = []

		# This level does not require any more clustering.
		# Because we care about leaves, we only create a mapping for these.
		if self.minClusterElements > len(identifiers) or level > self.numLevels:
			self.leavesToImageIndex[atCluster] = {}
			return

		self.kmModel.fit([self.descriptors[i] for i in identifiers])
		
		# This matrix contains a single list for each branch in the next level.
		# This way we can split the data into subsets and recursively k-means.
		nextLevelBranchIds = []
		for brId in range( self.branchingFactor ):
			nextLevelBranchIds.append([])

		# Get the cluster that belongs to each label, append them to their list.
		for identId in range( len(identifiers) ):
			nextLevelBranchIds[self.kmModel.labels_[identId]].\
				append(identifiers[identId])

		for brId in range( self.branchingFactor ):
			self.lastIndexedCluster += 1
			self.meanDescriptorClusters[self.lastIndexedCluster] = \
				self.kmModel.cluster_centers_[brId]
			self.indexedTree[atCluster].append(self.lastIndexedCluster)

			#Recursively proceed with the ith branch on the next level.
			self.updateIndexedTree(level + 1, 
				nextLevelBranchIds[brId], self.lastIndexedCluster)


	'''
	Update the inverted TF-IDF index with the descriptors from the data.
	For each leaf compute the count of matching descriptors for each file.
	Invert it, and keep it so we can use its keys for persistance.
	'''
	def setInvertedIndex(self, fileToDescriptorsMap):

		search = Vocabulary_Tree_Searcher(
			self.meanDescriptorClusters, self.indexedTree)

		for imagePath in fileToDescriptorsMap:

			descriptors = fileToDescriptorsMap[imagePath]
			for desc in descriptors:

				bestMatch = search.getBestLeafIdFromCluster(desc, 0)
				# Add a mapping for this file within this leaf.
				if imagePath not in self.leavesToImageIndex[bestMatch]:
					self.leavesToImageIndex[bestMatch][imagePath] = 0;
				#Update the count of words matching this file within this leaf.
				self.leavesToImageIndex[bestMatch][imagePath] += 1


		# Invert the mapping to make it file -[ leaves -> cnt
		for clusterId in self.leavesToImageIndex:
			desc = self.leavesToImageIndex[clusterId]
			
			for filePath in desc:
				
				if filePath in self.sourceToLeafIndex:
					self.sourceToLeafIndex[filePath][clusterId] = desc[filePath]
					continue

				self.sourceToLeafIndex[filePath] = {}


	'''
	Computes the entropy for each leaf cluster with elements.
	'''
	def computeLeafClusterEntropies(self):
		
		for clusterId in self.leavesToImageIndex:
			cluster = self.leavesToImageIndex[clusterId]
			if not cluster:
				self.leafClusterEntropies[clusterId] = 0
			else:
				self.leafClusterEntropies[clusterId] = \
					np.log(self.numDataPoints / len(cluster))



	'''
	Using the parameters, get the SIFT descriptors, and build the hierarchy.
	Use the branching factor of K and the maximum number of levels.
	'''
	def trainModel(self):

		print('BEGIN TRAINING')


		print('Step 0: Loading Training data.')
		filePaths = utils.getDataFileNames(
			self.trainingDataPath, self.numDataPoints)
		print('- Total Images Read:', len(filePaths))

		#Read a mapping from files to descriptors.
		#We use this for branching and for building the inverted index.
		fileToDescriptorsMap = self.siftExt.getFileToDescriptorsMap(filePaths)

		print('Step 1: Generating SIFT descriptors.')
		self.descriptors = np.array(\
			self.siftExt.extractAllDescriptors(fileToDescriptorsMap))
		print('- Total Descriptors:', self.descriptors.shape)


		print('Step 3: Building K-means hierarchy -> b-factor (%d), maxL (%d).' 
			% (self.branchingFactor, self.numLevels))
		#Mean of the root cluster.
		self.meanDescriptorClusters[0] = self.descriptors.mean(axis=0)
		descriptorEnumerations = [num for num in range(len(self.descriptors))]
		# At root, level 0, for all the descriptors.
		self.updateIndexedTree(0, descriptorEnumerations, 0)
		print('- Total Leaf Clusters:', len(self.leavesToImageIndex))


		print('Step 4: Updating the leaves with the corresponding descriptors.')
		self.setInvertedIndex(fileToDescriptorsMap)


		print('Step 5: Computing leaf cluster entropies.')
		self.computeLeafClusterEntropies()


		print('TRAINING COMPLETE')
		self.trainingComplete = True


	'''
	After training is complete, build a Vocabulary_Tree with the model params.
	'''
	def getTrainedModel(self):

		if(not self.trainingComplete):
			print('Training has not been performed yet! Call trainModel().')
			return

		return Vocabulary_Tree(
			self.leavesToImageIndex,
			self.sourceToLeafIndex,
			self.leafClusterEntropies,
			self.meanDescriptorClusters, 
			self.indexedTree)


	'''
	After training is complete, build a Vocabulary_Tree with the model params.
	'''
	def storeTrainedModel(self):

		model = self.getTrainedModel()
		if(not model):
			return

		utils.persistModel(model, self.savedModelDataPath)
		print('Trained model stored in:', self.savedModelDataPath)




###
# Class to store the model for a vocabulary tree.
###
class Vocabulary_Tree:

	def __init__(self, leavesToImageIndex, sourceToLeafIndex, 
			leafClusterEntropies, meanDescriptorClusters, indexedTree):
		self.leavesToImageIndex = leavesToImageIndex
		self.sourceToLeafIndex = sourceToLeafIndex
		self.leafClusterEntropies = leafClusterEntropies
		self.meanDescriptorClusters = meanDescriptorClusters
		self.indexedTree = indexedTree




###
# Class to facilitate lookups within a tree of clusters.
###
class Vocabulary_Tree_Searcher:
	def __init__(self, meanDescriptorClusters, indexedTree):
		self.meanDescriptorClusters = meanDescriptorClusters
		self.indexedTree = indexedTree


	'''
	Obtain the best match for a given descriptor among all the leaf means.
	Recursively match clusters to the descriptor until a leaf is reached.
	Use fromCluster to exclude sections not relevant to our search.
	'''
	def getBestLeafIdFromCluster(self, descriptor, fromCluster):

		bestClusterId = self.getBestLeafIdAtCluster(descriptor, fromCluster)
		
		return bestClusterId \
			if not self.indexedTree[bestClusterId] \
			else self.getBestLeafIdFromCluster(descriptor, bestClusterId)


	'''
	Obtain the best match for a given descriptor at the given level.
	We define this as the smallest euclidean distance.
	'''
	def getBestLeafIdAtCluster(self, descriptor, atCluster):

		minimumDistance, bestClusterId = np.inf, -1
		
		currentLevelClusters = self.indexedTree[atCluster]
		for cluster in currentLevelClusters:
			mean_vector = self.meanDescriptorClusters[cluster]
			dst = np.sqrt(np.sum((mean_vector - descriptor)**2))
			if(dst >= minimumDistance):
				continue
			minimumDistance = dst
			bestClusterId = cluster

		return bestClusterId
		

