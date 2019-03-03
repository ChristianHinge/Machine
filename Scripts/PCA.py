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


temp = np.array(["low"]*len(dOriginal["price"]))
boundNames = ["medium","high","very high"]
for b,bName in zip(bounds,boundNames):
    temp[dOriginal["price"].values>b] = bName

dOriginal["price-cat"] = temp



scatterData=dOriginal
scatterAttr=attNoK

for col in oneHotKDict:
    scatterData = scatterData.drop([col],axis=1)

#########Standard Statistics#############    
statMatrix=pd.DataFrame(data=np.zeros((17,3)),index=scatterAttr, columns=["Mean","Variance", "Standard Deviation"])
for col in scatterAttr:
     statMatrix["Mean"][col]=round(np.mean(scatterData[col].values),2)
     statMatrix["Variance"][col]=round(np.var(scatterData[col].values),2)
     statMatrix["Standard Deviation"][col]=round(np.std(scatterData[col].values),2)

######### Removal of ordinal dicrete attributes ############
attsToBeRemoved=["num-of-doors","num-of-cylinders","symboling"]
for col in attsToBeRemoved:
    scatterData = scatterData.drop([col],axis=1)
    scatterAttr.remove(col)    

#########Correlation Matrix#############
CorData=scatterData.drop(["price-cat"],axis=1)
CorMatrix=pd.DataFrame(data=np.corrcoef(CorData, rowvar=False),index=scatterAttr, columns=scatterAttr)

######### Scatter-Plot ############
if GENERATE_PLOTS:     

    classLabels = dOriginal["price-cat"].values
    classNames = sorted(set(classLabels))
    classDict = dict(zip(classNames,range(len(classNames))))
    y = np.array([classDict[value] for value in classLabels])
    M=len(scatterAttr)
    C=len(bounds)
    for m1 in range(M):
        plt.figure(figsize=(15,10))
        plt.suptitle('Correlation for: '+scatterAttr[m1])
        for m2 in range(M):              
            plt.subplot(M/2, 2, m2 + 1)
            for c in range(C):
                class_mask = (y==c)
                plt.plot(scatterData.values[class_mask,m2], scatterData.values[class_mask,m1], '.')
                plt.xlabel("Correlation with "+ scatterAttr[m2]+ ": "+str(round(CorMatrix.iloc[m1][m2],2)))
                plt.xticks([])
                plt.yticks([])        
        plt.legend(classNames)
        plt.savefig("../Figures/ScatterPlots/ScatterPlot"+scatterAttr[m1]+".png")    
    plt.show()
    
