import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import categoric2numeric, pickle

######### Loading Data #########

# We start by the path to the file that we're we need to load.
file_path = '../Data/pokemon.csv'

# The file is comma-seperated and there there is no header
data = pd.read_csv(file_path, sep=',')

######### Data Processing #########

#Changing booleans to binary
booleanAttr=["isLegendary","hasGender","hasMegaEvolution"]
for col in booleanAttr:
    data[col] = data[col].replace('False',0)
    data[col] = data[col].replace('True',1)



#Attributes that are to be k-encoded
toBeKencodedColNames = ['Type_1',"Type_2","Color","Egg_Group_1","Egg_Group_2","Body_Style"]

for col in toBeKencodedColNames:
     data[col] = data[col].fillna("None")

data["Pr_Male"] = data["Pr_Male"].fillna(np.nanmean(data["Pr_Male"].values))

dOriginal = data.copy()
data = data.drop("Name",axis = 1)
data = data.drop("Number", axis = 1)


#Log transformation
data["Weight_kg"]=np.log(data["Weight_kg"])
data["Height_m"]=np.log(data["Height_m"])
dLogTransform = data.copy()

#Data for Linear Regression
#dLinearReg=data.copy()

#One-out-of-K-encoding
attributeNames = list(data)
attrNames=attributeNames.copy()
dictK = {}

for colName in toBeKencodedColNames:
     X_num, attribute_names = categoric2numeric.categoric2numeric(data[colName])
     tempDataFrame = pd.DataFrame(data = X_num, columns=attribute_names)

     for i in range(len(attribute_names)):

          data[attribute_names[i]] = X_num[:,i]

     #Appending K-encoded columns
     dictK[colName] = attribute_names
     attrNames.append(colName)
#Delete old columns that have been k-encoded
for col in toBeKencodedColNames:
     data = data.drop([col],axis=1)
     #dLinearReg = dLinearReg.drop([col],axis=1)
     attributeNames.remove(col)
     
     


#Remove data withs NaNs
data = data.dropna(axis=0)
#dLinearReg=dLinearReg.dropna(axis=0)



#Normilization of one-out-of-K
#print(oneOutOfKColumns)
for key in dictK:
     k = len(dictK[key])

     for i in range(k):
          mu = np.mean(data[dictK[key][i]])
          sd = np.std(data[dictK[key][i]])

          
          if sd == 0:
               #Due to removal of NaN observations, some columns like Renault contain only 0. Remove these columns.
               data = data.drop(dictK[key][i],axis=1)
          else:    
               data[dictK[key][i]]=(data[dictK[key][i]]-mu)/(sd*k**0.5)

#Normilization of all other attributes
for attr in attributeNames:
     data[attr]=(data[attr]-np.nanmean(data[attr]))/(np.nanstd(data[attr]))

######### Data Export #########

#Export Data with normalized and one-out-of-k-encoded data

data.to_pickle("../Data/dNorm")
dOriginal.to_pickle("../Data/dOriginal")
dLogTransform.to_pickle("../Data/dLogTransform")
#dLinearReg.to_pickle("../Data/dLinearReg")


with open('../Data/1_hot_K_dict.pickle', 'wb') as handle:
    pickle.dump(dictK, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../Data/non_hot_k_list.pickle', 'wb') as handle:
    pickle.dump(attributeNames, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('../Data/allAttributes.pickle', 'wb') as handle:
    pickle.dump(attrNames, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(np.std(data.values,axis=0))


