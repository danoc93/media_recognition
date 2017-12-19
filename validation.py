'''
validation.py
Performs accuracy calculations to determine best branching factor (among other things).
'''

import os 
import json
import utils
import copy
import random
from vocabulary_tree import Tree_Builder
from image_retriever import Image_Retriever
import numpy as np

####### CONFIGURE ##########

BRANCHING_FACTORS = [10, 20, 30, 60]

# Read the following test set folders.
CURRENT_PATH =['TestSet1', 'TestSet2']
TEST_ROOT = CURRENT_PATH + '/Trainer/Data/'
TEST_DATA_POINTS = 100

#############################

# Load the configuration, and include the current path to make things faster.
CONFIGURATION = json.load(open(CURRENT_PATH + '/config.json'))
CONFIGURATION["projectPath"] = CURRENT_PATH

'''
Train the vocabulary_tree with the given branching factor.
Return an image retriever for that trained model.
'''
def trainForBranchingFactor(factor):

	config = copy.deepcopy(CONFIGURATION)
	config["branchingFactor"] = factor
	config["maxNumberOfLevels"] = factor + 1
	config["maxNumberOfKeyPoints"] = 700
	config["savedModelDataPath"] += str(factor)

	vtb = Tree_Builder(config)
	vtb.trainModel()
	vtb.storeTrainedModel()

	ir = Image_Retriever(config)
	ir.prepare()

	return ir


'''
Measures the accuracy for a single testPath.
'''
def measureAccuracy(image_retriever, testPath, maxNumOfMatches):
	
	p = TEST_ROOT + testPath
	files = utils.getDataFileNames(p, 100)
	files = random.sample(files, TEST_DATA_POINTS)


	print('*** Validating ', testPath)

	total = len(files)
	is_best_match = np.zeros((maxNumOfMatches, 1))
	is_best_match = np.squeeze(is_best_match)

	readf = 0
	for queryFile in files:
		if not readf % 5:
			print('Read files r->', readf)

		label = utils.getLabelFromFileName(queryFile)
		best_matches = image_retriever.findBestMatchingLabels(queryFile)
		if(label in best_matches):
			#This value is the same for each one after it.
			base = best_matches.index(label)
			for i in range(base, maxNumOfMatches):
				is_best_match[i] += 1
		readf += 1

	return is_best_match/len(files)


'''
Run a trial for each factor.
'''
for factor in BRANCHING_FACTORS:
	print('*** TEST: BRANCHING_FACTOR = %d\n' % factor)
	ir = trainForBranchingFactor(factor)
	best_n = ir.maxNumOfMatches

	total_best_match = np.zeros((best_n, 1))
	total_best_match = np.squeeze(total_best_match)

	numTests = len(TEST_PATHS)
	for test in TEST_PATHS:
		total_best_match += measureAccuracy(ir, test, best_n)

	total_best_match /= numTests

	print('AVG ACC TOP N = ', total_best_match)

	print('')
