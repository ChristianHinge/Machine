# exercise 11.4.1
import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors
from ProjectData import *
from matplotlib import pyplot as plt
# load data
#y = np.array(dNorm["isLegendary"])
#dNorm = dNorm.drop("isLegendary",axis = 1)
X = np.matrix(dNorm)

N, M = np.shape(X)
common_outliers = np.zeros(X.shape[0])
### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0**np.arange(-10,3))
logP = np.zeros(np.size(widths))
for i,w in enumerate(widths):
   print('Fold {:2d}, w={:f}'.format(i,w))
   density, log_density = gausKernelDensity(X,w)
   logP[i] = log_density.sum()
   
val = logP.max()
ind = logP.argmax()

width=widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X,width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1,)

# Plot density estimate of outlier score
figure(1)
bar(range(N),density[:N])
title('Density estimate')
plt.savefig("../Figures/densities.png")

# Possible outliers
print('Density estimate: Possible outliers')
for k in range(1,21):
    print('{0}. Pokemon Index: {1}'.format(k,i[k]))
    common_outliers[i[k]] += 1

### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1./(D.sum(axis=1)/K)

# Sort the scores
i = density.argsort()
density = density[i]

# Plot k-neighbor estimate of outlier score (distances)
figure(2)
bar(range(N),density[:N])
title('KNN density: Outlier score')
plt.savefig("../Figures/KNN_density.png")
# Possible outliers
print('KNN density: Possible outliers')
for k in range(1,21):
    print('{0}. Pokemon Index: {1}'.format(k,i[k]))
    common_outliers[i[k]] += 1


### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1./(D.sum(axis=1)/K)
avg_rel_density = density/(density[i[:,1:]].sum(axis=1)/K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]

# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(N),avg_rel_density[:N])
title('KNN average relative density: Outlier score')
plt.savefig("../Figures/KNN_ARD.png")
# Possible outliers
print('KNN average relative density: Possible outliers')
for k in range(1,21):
    print('{0}. Pokemon Index: {1}'.format(k,i_avg_rel[k]))
    common_outliers[i_avg_rel[k]] += 1


i = common_outliers.argsort()[::-1][0:40]

pokemonNames = np.array(dOriginal["Name"])
for score, pokemon in zip(common_outliers[i],i):
   print('{} Pokemon {} is marked as outlier {} times'.format(pokemon,pokemonNames[pokemon], score))
