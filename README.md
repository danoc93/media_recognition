# Media Classifier
Visual media recognition using a vocabulary tree and homographic projections.

# What does it do?
Given a set of training images, the system can be used to label unseen images, and thus be used for looking up data from databases and such. A simple program that uses the system has been integrated (simple_matcher.py).

The project can be used with other types of images, but all training-testing was performed using Stanford's media dataset:
https://exhibits.stanford.edu/data/catalog/rr389hv5603

For the first part, the following project attempts to implement the following paper:
http://www.cs.ubc.ca/~lowe/525/papers/nisterCVPR06.pdf

It also adds an extra layer of processing on top of the original implementation. It takes advantage of homographic projections to further analyse the list of prospective candidates.
Some context: https://en.wikipedia.org/wiki/Homography_(computer_vision)

# DEPENDENCIES
The project is built using Python 3.6, and assumes you have scikit-learn, numpy and opencv available as dependencies.

# HIGH-LEVEL DETAILS
0. Most configurable items (hyperparameters and preferences) can be changed from the config.json file.
1. The trainer can be used to create a model from the data you desire.
2. The retriever can be used to load a model and score an image against the trained database. It uses the L1 norm.
3. The homographic projection layer can be used to filter a list of matches and provide stronger predictions.
4. You can play around with the validation class to test different configurations!

# DISCLAIMER
1. The system may not be the most efficient speed wise (and I would recommend checking the scoring logic in case it is not implemented as specified). 
2. The modularization of the project allows its components to be treated/improved/adjusted independently, but it can always be improved!


# On simple_matcher.py
This little program uses a saved model to score an image against the database to get the top N matches, and then extracts the best match using homographic projections. The results are then placed on an HTML page and shown to the user.

To run: python simple_matcher.py (-t optional to train model using current configs) file_path_to_match



