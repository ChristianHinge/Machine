import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import categoric2numeric
from scipy.linalg import svd
import pickle

GENERATE_PLOTS = True

######## Loading Data ########

# We start by the path to the file that we're we need to load.
file_path = '../Data/car.data'

# The file is comma-seperated and there there is no header
data = pd.read_csv(file_path, sep=',', header=None)


# We manually type the attribute names
attributeNames = ["symboling", "normalized-losses", "make", "fuel-type","aspiration", "num-of-doors", 
                            "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", 
                            "curb-weight", "engine-type", "num-of-cylinders", "engine-size","fuel-system","bore","stroke", 
                            "compression-ratio","horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
# We assign the attribute names to the data columns
data.columns=attributeNames

#Removing normalized-losses due to substantial NaNs
data = data.drop(["normalized-losses"],axis=1)
attributeNames.remove("normalized-losses")

#Changing "?" with "NaN" and number-words with the actual value
for col in attributeNames:
    data[col] = data[col].replace('?','NaN')
    data[col] = data[col].replace('two',2)
    data[col] = data[col].replace('three',3)
    data[col] = data[col].replace('four',4)
    data[col] = data[col].replace('five',5)
    data[col] = data[col].replace('six',6)
    data[col] = data[col].replace('eight',8)
    data[col] = data[col].replace('twelve',12)

#Changing some of the numeric data types to integers/floats instead of strings
data["num-of-doors"] = pd.to_numeric(data["num-of-doors"],downcast='integer',errors='coerce')
data["bore"] = pd.to_numeric(data["bore"],errors='coerce')
data["stroke"] = pd.to_numeric(data["stroke"],errors='coerce')
data["peak-rpm"] = pd.to_numeric(data["peak-rpm"],downcast='integer',errors='coerce')
data["horsepower"] = pd.to_numeric(data["horsepower"],downcast='integer',errors='coerce')
data["price"] = pd.to_numeric(data["price"],downcast='integer',errors='coerce')

#Export Data with no NaN values
data.dropna().to_pickle("../Data/dOriginal")

#Attributes that are to be k-encoded
toBeKencodedColNames = ["make","fuel-type","aspiration","body-style","drive-wheels","engine-location","engine-type","fuel-system"]

#Extracting values of columns to be K-encoded.
car_names = np.unique(data["make"]).tolist()
fuel_types_names= np.unique(data["fuel-type"]).tolist()
aspiration_types=np.unique(data["aspiration"]).tolist()
body_style_names=np.unique(data["body-style"]).tolist()
drive_wheels_types=np.unique(data["drive-wheels"]).tolist()
engine_location_types=np.unique(data["engine-location"]).tolist()
engine_type_names=np.unique(data["engine-type"]).tolist()
fuel_system_types = np.unique(data["fuel-system"]).tolist()


#One Out of K Columns
dictK = {}
for colName in toBeKencodedColNames:
     X_num, attribute_names = categoric2numeric.categoric2numeric(data[colName])
     tempDataFrame = pd.DataFrame(data = X_num, columns=attribute_names)
     for i in range(len(attribute_names)):
          if np.isnan(X_num).any():
               print(attribute_names)
          data[attribute_names[i]] = X_num[:,i]
     dictK[colName] = attribute_names

#Delete columns
for col in toBeKencodedColNames:
     data = data.drop([col],axis=1)
     attributeNames.remove(col)

#Print data with NaNs 
#print("point 1", data.loc[data.isna().any(axis=1),data.isna().any(axis=0)])


#Remove data withs NaNs
data = data.dropna(axis=0)
plt.figure()
plt.hist(data["price"], density=True, histtype='step', cumulative=-1, label='Reversed emp.')
#plt.show()

####BOX-PLOT#######
if GENERATE_PLOTS:
     for col in attributeNames:
          plt.figure()
          plt.boxplot(data[col].values)
          plt.ylabel(col)
          plt.title('Car data set - boxplot')
          plt.savefig("../Figures/"+col+".png")


####SCATTER-PLOT###
#Creating data for scatter plots

if GENERATE_PLOTS: 
     dataForScatter=["num-of-doors","num-of-cylinders","symboling"]
     scatterData=data
     scatterAttr=attributeNames
     for col in dataForScatter:
          scatterData = scatterData.drop([col],axis=1)
          scatterAttr.remove(col)

     classLabels = data["num-of-cylinders"].values
     classNames = sorted(set(classLabels))
     classDict = dict(zip(classNames,range(len(classNames))))
     y = np.array([classDict[value] for value in classLabels])
     M=len(scatterAttr)
     C=len(classNames)
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
                         #plt.yticks([])
                    #ylim(0,X.max()*1.1)
                    #xlim(0,X.max()*1.1)
          plt.legend(classNames)

     plt.savefig("../Figures/ScatterPlot.png")

#Normilization of one-out-of-K
oneOutOfKColumns = [car_names , fuel_types_names, aspiration_types, body_style_names,drive_wheels_types,engine_location_types,engine_type_names,fuel_system_types]
#print(oneOutOfKColumns)
for att in oneOutOfKColumns:
     k = len(att)

     for i in range(k):
          mu = np.mean(data[att[i]])
          sd = np.std(data[att[i]])
          if sd == 0:
               data = data.drop(att[i],axis=1)
          else:    
               data[att[i]]=(data[att[i]]-k)/(sd*k**0.5)

#Normilization of all other attributes
for attr in attributeNames:
     data[attr]=(data[attr]-np.nanmean(data[attr]))/(np.nanstd(data[attr]))


#Save all column names
attributeNamesWithK=list(data)

#Export Data with normalized and one-out-of-k-encoded data
data.to_pickle("../Data/dNorm")

with open('../Data/1_hot_K_dict.pickle', 'wb') as handle:
    pickle.dump(dictK, handle, protocol=pickle.HIGHEST_PROTOCOL)



########## PCA ###########
if GENERATE_PLOTS:
     # PCA by computing SVD of Y
     U,S,V = svd(data,full_matrices=False)

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

