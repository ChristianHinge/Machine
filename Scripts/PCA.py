from CarsData import *
from matplotlib import pyplot as plt
from scipy.linalg import svd
from scipy import stats
import numpy as np

GENERATE_PLOTS = False

priceArray = dOriginal["price"].values
priceArray = np.sort(priceArray)
bounds = priceArray[::len(priceArray)//4][1:-1]
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

attsToBeRemoved=["num-of-doors","num-of-cylinders","symboling"]
scatterData=dOriginal
scatterAttr=attNoK
for col in attsToBeRemoved:
    scatterData = scatterData.drop([col],axis=1)
    scatterAttr.remove(col)

for col in oneHotKDict:
    scatterData = scatterData.drop([col],axis=1)

#########Correlation Matrix#############
CorData=scatterData.drop(["price-cat"],axis=1)
CorMatrix=pd.DataFrame(data=np.corrcoef(CorData, rowvar=False),index=scatterAttr, columns=scatterAttr)

######### Scatter-Plot ############
#dOriginal["price-cat"] = temp
if GENERATE_PLOTS: 
    #plt.figure()
    

   ####################################### 
    classLabels = dOriginal["price-cat"].values
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames,range(len(classNames))))
    y = np.array([classDict[value] for value in classLabels])
    M=len(scatterAttr)
    C=len(bounds)
    #plt.figure(figsize=(scatterData.shape))
    for m1 in range(M):
        plt.figure(figsize=(15,10))
        plt.suptitle('Correlation for: '+scatterAttr[m1])
        for m2 in range(M):              
            plt.subplot(M/2, 2, m2 + 1)
            for c in range(C):
                class_mask = (y==c)
                #print(class_mask)
                #print(m2)
                #print(data.values[np.array(class_mask),m2])
                plt.plot(scatterData.values[class_mask,m2], scatterData.values[class_mask,m1], '.')
                plt.xlabel(scatterAttr[m2])
                plt.xticks([])
                plt.yticks([])        
                #ylim(0,X.max()*1.1)
                #xlim(0,X.max()*1.1)
        plt.legend(classNames)
        plt.savefig("../Figures/ScatterPlots/ScatterPlot"+scatterAttr[m1]+".png")
        
    plt.show()
    
