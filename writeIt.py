'''
Created on 2 Aug 2013

@author: apuigdom
'''
from features import getStats
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor
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

    def translateArray(self, toChooseList, indexes, goal, size, indexRange):
        for i, index in enumerate(indexes[1:]):
            if i == goal:
                toChooseList[toChooseList >= indexes[i]] += indexRange
            else:
                toChooseList[toChooseList >= index-size] += size

    def getPredictionModels(self, sequences):
        print "Model "+str(0)
        model = pickle.load(open("models/models"+str(0)+".mod"))
        modelPredictions = model.predict(sequences)
        predictions = np.array([[pred] for pred in modelPredictions])
        for i in range(1,387):
            print "Model "+str(i)
            model = pickle.load(open("models/models"+str(i)+".mod"))
            modelPredictions = model.predict(sequences)
            modelPredictions = np.array([[pred] for pred in modelPredictions])
            predictions = np.concatenate((predictions,modelPredictions), axis=1)
        np.save("subpredictions", predictions)
        return predictions

    def getMainFeatures(self, train, indexes, goal, samples=1500):
        size = 300
        np.random.seed(goal+5000)
        beginning = indexes[goal]
        end = len(train) if goal == len(indexes) - 1 else indexes[goal + 1]
        print str(samples) + " sequences of " + str(goal)
        finalToChooseList = np.random.randint(beginning, end - size, size=samples)
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
        trainFeatures = self.getMainFeatures(self.train, indexes, 0)
        for i in range(1,len(indexes)):
            trainFeatures = np.concatenate((trainFeatures, self.getMainFeatures(self.train, indexes, i)))
        predictions = self.getPredictionModels(trainFeatures)
        finalArray = []
        for i in range(len(indexes)):
            modelPred = predictions[i*1500:(i+1)*1500]
            arrayPred = []
            for j in range(len(indexes)):
                arrayPred.append(np.mean(modelPred[:,j]))
            print arrayPred
            finalArray.append(arrayPred)
        finalArray = np.array(finalArray)
        np.save("finalArray", finalArray)
if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()