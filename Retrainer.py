'''
Created on 9 Aug 2013

@author: apuigdom
'''
from features import getStats
import numpy as np
import pandas as pd
from utils import loadModels
from sklearn.ensemble.forest import RandomForestRegressor
import pickle

class Retrainer:
    def __init__(self):
        print "Reading the training data..."
        self.train = pd.read_csv('train.csv')
        print "Reading models..."
        self.models = loadModels()    

    def getPredictions(self, sequences):
        print "Model "+str(0)
        model = self.models[0]
        modelPredictions = model.predict(sequences)
        predictions = np.array([[pred] for pred in modelPredictions])
        for i in range(1,387):
            print "Model "+str(i)
            model = self.models[i]
            modelPredictions = model.predict(sequences)
            modelPredictions = np.array([[pred] for pred in modelPredictions])
            predictions = np.concatenate((predictions,modelPredictions), axis=1)
        return predictions
    
    def getAttributes(self, train, indexes, goal, samples = 1500):
        np.random.seed(goal+1000)
        toChooseDevice = np.random.randint(len(indexes)-1, size=samples)
        toChooseDevice[toChooseDevice >= goal] += 1
        toChooseSeq = np.random.randint(samples, size=samples)
        sequenceGroup = [train[toChooseDevice[i]][toChooseSeq[i]] for i in range(samples)]
        sequenceGroup.extend(train[goal])
        trainVect = self.getPredictions(sequenceGroup)
        targetVect = [0] * (samples)
        targetVect.extend([1] * (samples))
        return (np.array(trainVect), np.array(targetVect))
    
    
    def getMainFeatures(self, train, indexes, goal, samples = 1500):
        size = 300
        np.random.seed(goal+1500)
        beginning = indexes[goal]
        end = len(train) if goal == len(indexes)-1 else indexes[goal+1]
        print str(samples) + " sequences of " + str(goal)
        toChoose = np.random.randint(beginning,end-size, size=samples)
        return np.array([getStats(train[i:i+size, :]) for i in toChoose])
    
    def run(self):
        print "Reading device separations..."
        indexes = np.load("indexesTrain.npy")
        self.train = self.train.values
        print "Getting attributes..."
        trainFeatures = [self.getMainFeatures(self.train,indexes,i) for i in range(len(indexes))]
        for i in range(len(indexes)):
            print i
            (trainVect, targetVect) = self.getAttributes(trainFeatures, indexes, i)
            classifier = RandomForestRegressor(n_estimators=500, verbose=2, n_jobs=1, random_state=1)
            classifier.fit(trainVect, targetVect)
            pickle.dump(classifier, open("remodels/models"+ str(i)+".mod", "w"))
            
if __name__ == "__main__":
    retrainer = Retrainer()
    retrainer.run()