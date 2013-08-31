import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trim_mean(x, trim=10):
    lower_bound = np.percentile(x, trim)
    upper_bound = np.percentile(x, (100-trim))
    return np.mean(x[(x>=lower_bound) & (x<=upper_bound)])
    
def getSimilarDevices(tol):
    data = pd.read_csv('train.csv')
    print "Getting sample rate...",
    ## Create steps
    data['T'] = data.groupby('Device').apply(lambda x: x['T'] - x['T'].shift(1)).fillna(207)
    ## Getting samples rate, then sort so that similar devices are close together
    data2 = data.groupby('Device')['T'].apply(lambda x: trim_mean(x))
    data2.sort()
    ## How many similar devices you'd want to include from the left and right
    ## Eg. If tol=3 you'll end up picking the next 3 devices with slightly higher
    ## samples rate AND the next 3 devices with slightly lower sample rate.
    similars = []
    for i,dev in enumerate(data2.index.values):
        begin=i-tol if (i-tol) >0 else 0
        end=i+1+tol if (i+tol+1) < len(data2) else len(data2)
        similars.append(
            (dev, list(data2.index.values[begin:i]) +
             list(data2.index.values[i+1:end]))
        )
    similars = dict(similars)
    return similars

if __name__ == "__main__":
    print "Reading the training data..."
    train = pd.read_csv('train.csv')
    getSimilarDevices(train, 3)