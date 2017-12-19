'''
This tool takes a at most 2 arguments. An image and optionally a -t flag.
If -t is set then we train a model with the current configurations.
We load a saved model and match the src image against the database.
The class updates an HTML template and requires it to be in the Result folder.
'''

import os 
from image_retriever import Image_Retriever
from vocabulary_tree import Tree_Builder
from homography import Homography_Finder
import json
import sys
import utils

n = len(sys.argv)
if n < 2 or n > 3 or (n == 3 and '-t' not in sys.argv):
	print('Invalid arguments! [required: filepath], [optional: -t]')
	exit()

train = len(sys.argv) == 3
queryFile = sys.argv[1]
if train and queryFile == '-t':
	queryFile = sys.argv[2]

# Load the configuration.
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIGURATION = json.load(open(CURRENT_PATH + '/config.json'))
CONFIGURATION["projectPath"] = CURRENT_PATH

#If required!
if train:
	vtb = Tree_Builder(CONFIGURATION)
	vtb.trainModel()
	vtb.storeTrainedModel()

# Load a trained model.
ir = Image_Retriever(CONFIGURATION)
ir.prepare()

if not ir.prepared:
	exit()

label = utils.getLabelFromFileName(queryFile)
best_matches = ir.findBestMatchingImages(queryFile)


print('Found TOP 10 matches!', len(best_matches))

# Get the best homography across the best matches!
print('Computing number of inliers!')
h = Homography_Finder(CONFIGURATION["kpThresholdHom"], 
	CONFIGURATION["maxNumberOfKeyPointsHom"],
	CONFIGURATION["scaleFactor"])

bestMatchHomography = ''
maxMask, maxGood, maxim = None, None, 0
for image in best_matches:
	i, m, g = h.findHomography(queryFile, image)
	if i > maxim:
		maxim, bestMatchHomography = i, image

top_match_vt = best_matches[0]
top_match_vtpi = bestMatchHomography

print('Best matches <VT>, <VT+I>:', top_match_vt, top_match_vtpi)

if not top_match_vtpi:
	print ('Could not find enough inliers in any of the matches!')
	exit()

# Pretty silly but simple way of changing the images used for the HTML file.
from shutil import copyfile
for i in range(len(best_matches)):
	copyfile(best_matches[i], 'Result/T'+str(i))

copyfile(queryFile, 'Result/target')
copyfile(top_match_vt, 'Result/TMVT')
copyfile(top_match_vtpi, 'Result/TMVTI')

os.system('open Result/result.html')


