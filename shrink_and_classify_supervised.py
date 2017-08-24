# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:09:10 2017

@author: mbell
"""

############################
# Load libraries - should all be available in Anaconda
############################
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.misc import imresize

from sklearn.metrics import confusion_matrix

from sklearn.cross_validation import train_test_split
from shutil import copyfile

##################################################
# Change folder names to suit
# One folder per category of form
##################################################
form_types = ['Declaration','Enrolment','Discharge','FrontSheets','Miscellaneous','Casualty','Questions','Statement']

training_files = set()
training_features = []     # List of training vectors
training_class = []        # List of classifications (form type) for each vector
training_file_count = {}   # Limits number of training examples
class_limit = 20   # Lower this for testing purposes. It will only train on this number of samples per form type.

########################################################
# For each type of form:
# Look in training example folder,
# Open each training image as grey scale
# Shrink it to a small size (experiment with size to get best results)
# Convert to a vector, and add to training set.
########################################################
for form in form_types:
    training_file_count[form] = 0
    file_list = set(os.listdir('.\\0\\Images\\OriginalClassified\\' + form))
    file_class = form
    for f in file_list:
        training_files.add(f)
        if training_file_count[file_class] == class_limit:
            continue
        training_file_count[file_class] += 1    
        
        img = io.imread("0\\Images\\Originals\\" + f, as_grey = True)
        
        shrunken = imresize(img, (20,10)).reshape((1,(20*10)))
    
        training_features.append(shrunken.tolist()[0])
        training_class.append(file_class)

###############################################
# Create training/test split
# Choose learning algorithm - SVM in this case
###############################################
print(training_file_count)
training_features = np.array(training_features)
X_train, X_test, y_train, y_test = train_test_split(training_features, training_class, test_size = 0.2, random_state=0)
from sklearn.svm import SVC
estimator = SVC(kernel='linear')

#######################################################################################################
# Cross validation allows many tests to be run with different portions of the training data to be run
# It just gives a better idea of how well the model is performing
#######################################################################################################
from sklearn.cross_validation import ShuffleSplit
cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
from sklearn.grid_search import GridSearchCV
import numpy as np
gammas = np.logspace(-6, -1, 10)
classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
classifier.fit(X_train, y_train)
#plt.show()

###############################################################
# Test the classifier by running it against the test data set
# The confusion matrix shows where it is going right and wrong
###############################################################
print("Score:",classifier.score(X_test, y_test))
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))

###############################################################
# Use the trained classifier against some real unclassified files
# For each one predict a category and copy it to the appropriate folder
# auto_limit variable can be set to limit the size of the test
###############################################################
auto_limit = 200
auto_count = 0
unseen_files = set(os.listdir('.\\0\\Images\\To_Classify\\'))
for f in unseen_files:
    if f in training_files:
        print(f)
        continue
    img = io.imread('.\\0\\Images\\To_Classify\\' + f, as_grey = True)        
    shrunken = imresize(img, (20,10)).reshape((1,(20*10)))
    pred = classifier.predict(shrunken)
    copyfile('.\\0\\Images\\To_Classify\\' + f, '.\\0\\Images\\AutoClassified\\' + pred[0] + '\\' + f)
    auto_count += 1
    if auto_count > auto_limit:
        break