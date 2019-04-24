# exercise 11.4.1
import numpy as np
from matplotlib.pyplot import (figure, imshow, bar, title, xticks, yticks, cm,
                               subplot, show)
from scipy.io import loadmat
from toolbox_02450 import gausKernelDensity
from sklearn.neighbors import NearestNeighbors
from ProjectData import *

# load data
X = np.matrix(dNorm)
N, M = np.shape(X)




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
show()
# Possible outliers
print('Density estimate: Possible outliers')
for k in range(1,21):
    print('{0}. Pokemon Index: {1}'.format(k,i[k]))


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
show()
# Possible outliers
print('KNN density: Possible outliers')
for k in range(1,21):
    print('{0}. Pokemon Index: {1}'.format(k,i[k]))



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
show()
# Possible outliers
print('KNN average relative density: Possible outliers')
for k in range(1,21):
    print('{0}. Pokemon Index: {1}'.format(k,i_avg_rel[k]))



### Distance to 5'th nearest neighbor outlier score
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

# Outlier score
score = D[:,K-1]
# Sort the scores
i = score.argsort()
score = score[i[::-1]]

# Plot k-neighbor estimate of outlier score (distances)
figure(4)
bar(range(N),score[:N])
title('5th neighbor distance: Outlier score')
show()
# Possible outliers

print('5th neighbor distance: Possible outliers')
for k in range(1,21):
    print('{0}. Pokemon Index: {1}'.format(k,i[k]))
#    

