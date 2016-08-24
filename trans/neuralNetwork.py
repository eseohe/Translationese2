'''
Created on Aug 24, 2016

@author: admin
'''
from trans.classifier import classifier
import scipy

import random
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import os
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier



from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score
from sklearn.neural_network import BernoulliRBM


class neuralNetwork(classifier):
    '''
    classdocs
    '''
    activation='relu'
    algorithm='l-bfgs'
    alpha=1e-05
    batch_size='auto'
    beta_1=0.9
    beta_2=0.999
    early_stopping=False
    epsilon=1e-08
    hidden_layer_sizes=(5, 2)
    learning_rate='constant',
    learning_rate_init=0.001; max_iter=200; momentum=0.9
    nesterovs_momentum=True; power_t=0.5; random_state=1; shuffle=True
    tol=0.0001; validation_fraction=0.1; verbose=False
    warm_start=False

    def __init__(self, X, y):
        '''
        Constructor
        ''' 
        classifier.__init__(self, X, y)
        
    

    def splitTrainTest(self):
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = cross_validation.train_test_split(self.X, self.y, 
                                                                                            test_size=self.splitPercent,
                                                                                            random_state=0)
                                                                                            
    def train(self):
        
        if self.n_foldCV <= 0:
            print ('No cross validation required. If required set the parameter to a positive number')
            clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
            clf.fit(self.Xtrain, self.ytrain) 
        else:
            #not yet implemented, so does the same thing as without cv                  
            clf.fit(self.Xtrain, self.ytrain)
            
        self.model = clf
            
            
    def saveModel(self, location, saveAs):
        if os.path.exists(location):
            with open(location+'/'+saveAs, 'wb') as output:
                pickle.dump(self.model, output, pickle.HIGHEST_PROTOCOL)
        else:
            print ('Please enter a valid directory')
            
    def loadModel(self, location, savedAs):
        if os.path.exists(location):
            if os.path.isfile(location+'/'+savedAs):
                with open(location+'/'+savedAs, 'rb') as input:
                    self.model = pickle.load(input)
            else:
                print ('file does not exist')
        else:
            print ('Please enter a valid directory')
    

    def predict(self):
        return self.model.predict(self.Xtest)
        
        
        
        
    def evaluate(self):
        y_pred = self.predict()
        print ('Accuracy: ', accuracy_score(self.ytest, y_pred))
        print ('Precision: ', average_precision_score(self.ytest, y_pred))
        print ('Recall: ', recall_score(self.ytest, y_pred))
        print ('F-score: ', f1_score(self.ytest, y_pred))
        