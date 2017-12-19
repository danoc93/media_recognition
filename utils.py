import glob
import pickle

'''
Get a list of full paths to each of the data images.
'''
def getDataFileNames(dataPath, maxNumFiles = 0):
	
	filePaths = glob.glob(dataPath + '/*.jpg')
	filePaths.sort()

	if(maxNumFiles > 0):
		filePaths = filePaths[:maxNumFiles]

	return filePaths


'''
Get a label 'XXX from a file named 'XXX.jpg'.
'''
def getLabelFromFileName(fileName):
	return (fileName.split('/')[-1]).split('.')[0]


'''
Use pickle to persist a trained tree model.
'''
def persistModel(vocabularyTree, filePath):

	outputFile = open(filePath, "wb")
	pickle.dump(vocabularyTree, outputFile)
	outputFile.close()


'''
Use pickle to load a trained tree model.
'''
def loadModel(filePath):

	inputFile = open(filePath, "rb")
	vocabularyTree = pickle.load(inputFile)
	inputFile.close()
	return vocabularyTree