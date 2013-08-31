'''
Created on 9 Aug 2013

@author: apuigdom
'''
import pickle
import numpy as np


def loadModels():
    models = [pickle.load(open("models/models"+str(i)+".mod")) for i in range(387)]
    for model in models:
        model.n_jobs = 1
    return models
def loadReModels():
    return [pickle.load(open("remodels/models"+str(i)+".mod")) for i in range(387)]
def getLookupTable(devices):
    lookupTable = np.zeros(1038)
    for index in range(len(devices)):
        lookupTable[devices[index]] = index
    return lookupTable