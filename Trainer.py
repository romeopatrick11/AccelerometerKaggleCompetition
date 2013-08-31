'''
Created on 2 Aug 2013

@author: apuigdom
'''
from features import getStats
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
from Predictor2 import Predictor2

class Trainer:
    def __init__(self):
        print "Reading the training data..."
        self.train = pd.read_csv('train.csv')

    def getSeparationIndexes(self, train):
        index = 0
        sequence = 0
        indexList = []
        for element in range(len(train)):
            if train['Device'][element] != sequence:
                index = element
                indexList.append(index)
                print index
                sequence = train['Device'][element]
        return np.array(indexList)

    def getSeparationDevices(self, train):
        index = 0
        sequence = 0
        indexList = []
        for element in range(len(train)):
            if train['Device'][element] != sequence:
                index = train['Device'][element]
                indexList.append(index)
                print index
                sequence = train['Device'][element]
        return np.array(indexList)

    def getTrainData(self, splittedData, i):
        toReturn = np.delete(splittedData, i, 0)
        toReturn = tuple(tuple(x) for x in toReturn)
        toReturn = np.concatenate(toReturn)
        return toReturn

    def getAttributes(self, train, indexes, goal, samples=5000):
        targetVect = [1] * (samples)
        targetVect.extend([0] * (samples))
        trainVect = train[goal]
        return (trainVect, np.array(targetVect))

    def translateArray(self, toChooseList, indexes, goal, size, indexRange):
        for i, index in enumerate(indexes[1:]):
            if i == goal:
                toChooseList[toChooseList >= indexes[i]] += indexRange
            else:
                toChooseList[toChooseList >= index-size] += size

    def getMainFeatures(self, train, indexes, goal, samples=5000):
        size = 300
        np.random.seed(goal)
        beginning = indexes[goal]
        end = len(train) if goal == len(indexes) - 1 else indexes[goal + 1]
        print str(samples) + " sequences of " + str(goal)
        toChooseGoal = np.random.randint(beginning, end - size, size=samples)
        print str(samples) + " sequences of other stuff than " + str(goal)
        toChooseNotGoalRange = len(train) - size*(len(indexes)-1) - end + beginning
        toChooseNotGoal = np.random.randint(0, toChooseNotGoalRange, size=samples)
        self.translateArray(toChooseNotGoal, indexes, goal, size, end-beginning)
        finalToChooseList = np.concatenate((toChooseGoal, toChooseNotGoal))
        mapArray = np.array([train[i:i + size, :] for i in finalToChooseList])
        print "Calculating stats for " + str(goal)
        mainFeatures = getStats(mapArray)
        print "Done with " + str(goal)
        return mainFeatures

    def run(self):
        print "Reading device separations..."
        indexes = np.load("indexesTrain.npy")
        self.train = self.train.values
        print "Getting attributes..."
        trainFeatures = [self.getMainFeatures(self.train, indexes, i) for i in range(len(indexes))]
        for i in range(len(indexes)):
            (trainVect, targetVect) = self.getAttributes(trainFeatures, indexes, i)
            classifier = RandomForestRegressor(n_estimators=500, verbose=2, n_jobs=4, random_state=1)
            classifier.fit(trainVect, targetVect)
            pickle.dump(classifier, open("models/models" + str(i) + ".mod", "w"))

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
    predictor = Predictor2()
    predictor.run()