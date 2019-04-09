from ProjectData import *
from matplotlib import pyplot as plt
from scipy.linalg import svd
from scipy import stats
import numpy as np

#Classification for later use
"""
priceArray = dOriginal["price"].values
priceArray = np.sort(priceArray)
bounds = priceArray[::len(priceArray)//4][1:-1]
plt.plot(priceArray,'.r')
for b in bounds:
    plt.plot([1,len(priceArray)],[b,b],'k--')
"""

# PCA by computing SVD of Y
U,S,V = svd(dNorm,full_matrices=False)

######### Variance explained #########
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
plt.savefig("../Figures/ExplainedVariance.png")

######### The first 3 principle components #########
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = 0.8
f, handles = plt.subplots(1,3,figsize=(10,5))
for i in pcs:
    plt.subplot(1,3,i+1)
    noAtt = 16
    r = np.arange(noAtt)
    indices = np.abs(V[:,i]).argsort()[-noAtt:][::-1]    
    plt.bar(r+1-(1-bw), V[:,i][indices], width=bw,align='center')
    xlabels = np.array(list(dNorm))[indices]
    plt.xticks(r+bw, xlabels,rotation = 'vertical')
    if i==0:
        plt.ylabel('Component coefficients')
    plt.grid()
    plt.title(f"PCA Component {i} Coefficients")
plt.gcf().subplots_adjust(bottom=0.3)
plt.savefig("../Figures/PCcoefficients.png")


######### Data projection on first 3 principle components #########

# Project the centered data onto principal component space
Z = dNorm.values @ V

# Plot PCA of the data
fig = plt.figure(figsize=(10,5))
#Z = array(Z)
fig.suptitle("PCA Projection",size=20)
plt.subplot(1,2,1)
plt.plot(Z[:,0], Z[:,1], 'o', alpha=.5)
plt.xlabel('PC 1',fontsize=14)
plt.ylabel('PC 2',fontsize=14)

plt.subplot(1,2,2)
plt.plot(Z[:,0], Z[:,2], 'o', alpha=.5)
plt.xlabel('PC 1',fontsize=14)
plt.ylabel('PC 3',fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('../Figures/PCprojection.png')
