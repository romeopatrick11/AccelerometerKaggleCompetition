'''
Created on 3 Aug 2013

@author: apuigdom
'''
import numpy as np
import pandas as pd
from features import getStats
import pickle
class Predictor2:
    def getLookupTable(self, devices):
        lookupTable = np.zeros(1038)
        for index in range(len(devices)):
            lookupTable[devices[index]] = index
        return lookupTable
    
    def getTestStats(self):
        sequence = 0
        indexSeq = 0
        index = 0
        print "Reading csv"
        test = pd.read_csv('test.csv')
        test = test.values
        arrayTest = []
        for element in range(len(test)):
            if test[element][4] != sequence:
                if sequence != 0:
                    print "Sequence "+str(index)
                    arrayTest.append(test[indexSeq:element])
                    indexSeq = element
                    index += 1
                sequence = test[element][4]
                
        print "Sequence "+str(index)
        arrayTest.append(test[indexSeq:len(test)])
        print "Calculating stats"
        stats = getStats(np.array(arrayTest))
        np.save("testStats", stats)
        return stats
    
    def writePredictions(self, predictions):
        questions = pd.read_csv('questions.csv')
        idq = questions['QuestionId']
        results = open('submission.csv','w')
        results.write('QuestionId,IsTrue\n')
        for i in range(len(idq)):
            results.write(str(idq[i]) + "," + str(predictions[i]))
            results.write('\n')
    #Read questions
    
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
        

    def getProbDevices(self, indexes):
        print "Calculating indexes.."
        train = pd.read_csv('train.csv')
        lengthTrain = len(train)
        del(train)
        indexes = np.load("indexesTrain.npy")
        numDevices = np.zeros(len(indexes))
        numDevices[0:len(indexes)-1] = np.diff(indexes)
        numDevices[len(indexes)-1] = lengthTrain-indexes[len(indexes)-1]
        numDevices = numDevices/lengthTrain
        return numDevices
    
    
    def run(self, calculateStats=False, calculatePredictions=False):
        print "Loading test data..."
        sequences = np.load("testStats.npy") if not calculateStats else self.getTestStats()
        print "Loading devices..."
        devices = np.load("devices.npy")
        print "Getting lookup device table..."
        devicesIndexes = self.getLookupTable(devices)
        print "Getting devices probabilities"
        print "Loading questions..."
        questionFile = open("questions.csv","r")
        questions = questionFile.readlines()
        print "Getting intermediate predictions..."
        predictions = np.load("subpredictions.npy") if not calculatePredictions else self.getPredictionModels(sequences)
        print "Going for the final predictions..."
        finalPredictions = []
        for element in range(len(predictions)):
                index = element + 1
                print "Sequence "+str(index)
                questionsLine = questions[index].split(',')
                modelNum = devicesIndexes[int(questionsLine[2])]
                predictionModels = predictions[element]
                prediction = predictionModels[modelNum]
                print "Prediction "+str(prediction)
                finalPredictions.append(prediction)
        newPredictions = np.array(finalPredictions)
        np.save("predictions", newPredictions)
        self.writePredictions(newPredictions)
        
if __name__ == "__main__":
    predictor = Predictor2()
    predictor.run()
