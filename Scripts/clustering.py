# exercise 10.2.1
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from ProjectData import *
import numpy as np

# Load Matlab data file and extract variables of interest
dNorm = dNorm.drop("isLegendary",axis=1)
X = np.array(dNorm)
y = np.array(dOriginal["isLegendary"])
attributeNames = list(dNorm)
N, M = X.shape
C = 2


# Perform hierarchical/agglomerative clustering on data matrix
Method = 'complete'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 2
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()

print('Ran Exercise 10.2.1')