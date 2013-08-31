
'''
Created on 3 Aug 2013

@author: apuigdom
'''
import numpy as np
from scipy.stats.stats import skew, kurtosis, ss, sem
from scipy.stats.mstats_basic import mquantiles
import decimal

def getFourMoments(sequence, ax=1):
    finalArray = [np.mean(sequence, axis=ax), np.var(sequence, axis=ax), 
                  skew(sequence, axis=ax), kurtosis(sequence, axis=ax), 
                  sem(sequence, axis=ax)]
    if ax != None:
        finalArray = np.array(finalArray)
        finalArray = finalArray.T
        return np.concatenate((finalArray, np.array(mquantiles(sequence, axis=ax))),axis=ax)
    finalArray.extend(mquantiles(sequence, axis=ax))
    return np.array(finalArray)


def pearsonr(x, y):
    # x and y should have same length.
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    mx = x.mean()
    my = y.mean()
    xm, ym = x-mx, y-my
    r_num = n*(np.add.reduce(xm*ym))
    r_den = n*np.sqrt(ss(xm)*ss(ym))
    r = (r_num / r_den)
    r = max(min(r, 1.0), -1.0)
    return r

def getResolution(number):
    return len(str(number))

def getTimestampFeatures(seq):
    trim = 10
    lowerBound = np.percentile(seq, trim)
    upperBound = np.percentile(seq, (100-trim))
    return getFourMoments(seq[(seq>=lowerBound) & (seq<=upperBound)], ax=None)


def getResolutionFeatures(seq):
    seq = seq[seq > 2]
    return [np.mean(seq), np.var(seq)]

def getStats(sequence):
    timestamps = sequence[:,:,0]
    difference = np.diff(timestamps)
    print "Timestamp features..."
    finalArray = np.array([getTimestampFeatures(seq) for seq in difference])
    X = sequence[:,:,1]
    Y = sequence[:,:,2]
    Z = sequence[:,:,3]
    print "Raw features..."
    finalArray = np.concatenate((finalArray,getFourMoments(X),getFourMoments(Y), 
                                 getFourMoments(Z)), axis=1)
    
    getResolutionSeq = np.vectorize(getResolution)
    print "Resolution..."
    resX = getResolutionSeq(np.abs(X))
    resY = getResolutionSeq(np.abs(Y))
    resZ = getResolutionSeq(np.abs(Z))
    auxArrayX = np.array([getResolutionFeatures(seq) for seq in resX])
    auxArrayY = np.array([getResolutionFeatures(seq) for seq in resY])
    auxArrayZ = np.array([getResolutionFeatures(seq) for seq in resZ])
    finalArray = np.concatenate((finalArray,auxArrayX,auxArrayY, 
                                 auxArrayZ), axis=1)
    norm = np.sqrt(np.square(X)+np.square(Y)+np.square(Z))
    finalArray = np.concatenate((finalArray, getFourMoments(norm)), axis=1)
    
    #angleA = np.arctan2(Y, X)
    #angleB = np.arctan2(Z, X)
    #angleC = np.arctan2(Y, Z)
    #finalArray.extend([np.mean(angleA), np.mean(angleB), np.mean(angleC), 
    #                   np.var(angleA), np.var(angleB), np.var(angleC)])
    print "Pearsons..."
    pearson = np.array([[pearsonr(x, y), pearsonr(x, z), pearsonr(y, z)] for x, y, z in zip(X, Y, Z)])
    finalArray = np.concatenate((finalArray,pearson), axis=1)
    return finalArray
