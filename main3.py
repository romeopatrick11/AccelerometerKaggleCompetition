import matplotlib.pyplot as plt
import numpy as np
from cluster_similar_device import getSimilarDevices
finalArray = np.load("finalArray.npy")
row = finalArray[1]
row2 = np.where(row > 0.05)[0]
print row2
row = finalArray[1].T
row2 = np.where(row > 0.05)[0]
print row2
"""finalArray[np.diag_indices_from(finalArray)] = 0
plt.pcolor(finalArray)
plt.show()
# For every ground truth, what is the maximum
GTM = np.max(finalArray,axis=0)
# For every model, what its highest mark
MM = np.max(finalArray,axis=1)
print GTM
print MM"""

dictionary = getSimilarDevices(10)
for index, key in enumerate(dictionary.keys()):
    print str(index)+" "+str(key)
print dictionary