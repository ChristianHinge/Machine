# exercise 10.2.1
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from ProjectData import *
import numpy as np
from toolbox_02450 import clusterval
from matplotlib import pyplot as plt
# Load Matlab data file and extract variables of interest
print(list(dNorm))
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

print(attributeNames)
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
dendrogram(Z, truncate_mode='level', p=max_display_levels,color_threshold=10,leaf_font_size=10,no_labels=True) #labels=np.array(dOriginal["Name"])
show()

Rand, Jaccard, NMI = clusterval(y,b)    
print(Rand)
print(Jaccard)
print(NMI)

print(b)
legendaryGroups = b[dOriginal["isLegendary"]==True]
for i in range(4):
    print("{} percent of group {} is legendary".format(round(sum(legendaryGroups == i+1)/sum(b==i+1)*100,2),i+1))
    print(sum(b==i+1))


mean_data = np.zeros((dNorm.shape[1],4))

for i in range(X.shape[1]):
    X[:,i] = X[:,i] / np.std(X[:,i])

for i in range(4):
    mean_data[:,i] = np.mean(np.array(X[(i+1) == b,:]),axis=0)

intervals = list(range(0,dNorm.shape[1],23))+[dNorm.shape[1]]

figs, axs = plt.subplots(nrows = 1, ncols = 3)
figs.set_figheight(15)
figs.set_figwidth(15)
for i in range(len(intervals)-1):
    lower = intervals[i]
    upper = intervals[i+1]
    #print(type(axs))
    
    #col = i%3
    ax = axs[i]
    ax.matshow(mean_data[lower:upper])
    ax.set_xticklabels(["","1","2","3","4"])
    ax.set_yticks(range(23))
    ax.set_yticklabels(attributeNames[lower:upper])

figs.suptitle("Hirearchiral Clustering",fontsize=30)
#figs.tight_layout()
plt.savefig("../Figures/Means.png")