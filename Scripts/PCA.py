from CarsData import *
from matplotlib import pyplot as plt
from scipy.linalg import svd
import numpy as np

GENERATE_PLOTS = True

priceArray = dOriginal["price"].values
print(dOriginal)
priceArray = np.sort(priceArray)
bounds = priceArray[::len(priceArray)//4][1:-1]
print(bounds)
plt.plot(priceArray,'.r')
for b in bounds:
    plt.plot([1,len(priceArray)],[b,b],'k--')
#plt.show()


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
#plt.show()



temp = np.array(["low"]*len(dOriginal["price"]))
boundNames = ["medium","high","very high"]
for b,bName in zip(bounds,boundNames):
    temp[dOriginal["price"].values>b] = bName

dOriginal["price-cat"] = temp

#dOriginal["price-cat"] = temp
if GENERATE_PLOTS: 
    plt.figure()
    attsToBeRemoved=["num-of-doors","num-of-cylinders","symboling"]
    scatterData=dOriginal
    scatterAttr=attNoK
    for col in attsToBeRemoved:
        scatterData = scatterData.drop([col],axis=1)
        scatterAttr.remove(col)

    for col in oneHotKDict:
        scatterData = scatterData.drop([col],axis=1)

    classLabels = dOriginal["price-cat"].values
    classNames = sorted(set(classLabels))
    print(classLabels)
    classDict = dict(zip(classNames,range(len(classNames))))
    y = np.array([classDict[value] for value in classLabels])
    M=len(scatterAttr)
    C=len(bounds)
    plt.figure(figsize=(scatterData.shape))
    for m1 in range(M):
        for m2 in range(M):
            plt.subplot(M, M, m1*M + m2 + 1)
            for c in range(C):
                class_mask = (y==c)
                #print(class_mask)
                #print(m2)
                #print(data.values[np.array(class_mask),m2])
                plt.plot(scatterData.values[class_mask,m2], scatterData.values[class_mask,m1], '.')
                if m1==M-1:
                        plt.xlabel(scatterAttr[m2])
                else:
                        plt.xticks([])
                if m2==0:
                        plt.ylabel(scatterAttr[m1])
                else:
                        plt.yticks([])
                #ylim(0,X.max()*1.1)
                #xlim(0,X.max()*1.1)
        plt.legend(classNames)
    plt.show()
    plt.savefig("../Figures/ScatterPlot.png")
