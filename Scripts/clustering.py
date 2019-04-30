# exercise 10.2.1
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from ProjectData import *
import numpy as np
from toolbox_02450 import clusterval

# Load Matlab data file and extract variables of interest
dNorm = dNorm.drop("isLegendary",axis=1)


dNorm = dNorm[dOriginal["Name"] != "Shuckle"]
dNorm = dNorm[dOriginal["Name"] != "Chansey"]
dNorm = dNorm[dOriginal["Name"] != "Blissey"]
dNorm = dNorm[dOriginal["Name"] != "Ditto"]

dOriginal = dOriginal[dOriginal["Name"] != "Shuckle"]
dOriginal = dOriginal[dOriginal["Name"] != "Chansey"]
dOriginal = dOriginal[dOriginal["Name"] != "Blissey"]
dOriginal = dOriginal[dOriginal["Name"] != "Ditto"]



X = np.array(dNorm)
y = np.array(dOriginal["isLegendary"])

print(dNorm.shape)
print(dOriginal.shape)

attributeNames = list(dNorm)
N, M = X.shape
C = 2


# Perform hierarchical/agglomerative clustering on data matrix #Weighted, og Complete virker bedste
Method = 'complete'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 4

b = fcluster(Z, criterion='maxclust', t=Maxclust)
print(len(b))
figure(1)
clusterplot(X, b.reshape(b.shape[0],1), y=y)

# Display dendrogram
max_display_levels=20
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels,color_threshold=10, labels=np.array(dOriginal["Name"]),leaf_font_size=10)
show()

Rand, Jaccard, NMI = clusterval(y,b)    
print(Rand)
print(Jaccard)
print(NMI)
print('Ran Exercise 10.2.1')


legendaryGroups = b[dOriginal["isLegendary"]==True]
for i in range(4):
    print("{} percent of group {} is legendary".format(round(sum(legendaryGroups == i+1)/sum(b==i+1)*100,2),i+1))
    print(sum(b==i+1))
