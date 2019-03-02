from CarsData import *
from matplotlib import pyplot as plt
from scipy.linalg import svd
import numpy as np


print(dNorm)
# PCA by computing SVD of Y
U,S,V = svd(dNorm,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 
threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
