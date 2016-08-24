'''
Created on Aug 23, 2016

@author: admin
'''
from trans.classifier import classifier
import scipy

import random
import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import os
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score



class RandomForest(classifier):
    
    
    n_estimators=20
      
    
    
    def __init__(self, X, y):
        classifier.__init__(self, X, y)
    
    def splitTrainTest(self):
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = cross_validation.train_test_split(self.X, self.y, 
                                                                                            test_size=self.splitPercent,
                                                                                            random_state=0)
                                                                                                
        
    def train(self):
        
        if self.n_foldCV <= 0:
            print ('No cross validation required. If required set the parameter to a positive number')
            clf = RandomForestClassifier(n_estimators=20)
            clf.fit(self.Xtrain, self.ytrain)
        else:
            
            # Set the parameters by cross-validation
            # specify parameters and distributions to sample from
            param_dist = {"max_depth": [3, None],
                          "max_features": random.randint(1, 11),
                          "min_samples_split": random.randint(1, 11),
                          "min_samples_leaf": random.randint(1, 11),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
                    
                    
            n_iter_search = 20
            clf = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
            
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
