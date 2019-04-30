from apyori import apriori
import numpy as np
from similarity import binarize2
from ProjectData import *
# This is a helper function that transforms a binary matrix into transactions.
# Note the format used for courses.txt was (nearly) in a transaction format,
# however we will need the function later which is why we first transformed
# courses.txt to our standard binary-matrix format.

def mat2transactions(X, labels=[]):
    T = []
    for i in range(X.shape[0]):
        l = np.nonzero(X[i, :])[0].tolist()
        if labels:
            l = [labels[i] for i in l]
        T.append(l)
    return T

# This function print the found rules and also returns a list of rules in the format:
# [(x,y), ...]
# where x -> y
def print_apriori_rules(rules):
    frules = []
    for r in rules:
        for o in r.ordered_statistics:        
            conf = o.confidence
            supp = r.support
            x = ", ".join( list( o.items_base ) )
            y = ", ".join( list( o.items_add ) )
            print("{%s} -> {%s}  (supp: %.3f, conf: %.3f)"%(x,y, supp, conf))
            frules.append( (x,y) )
    return frules

# We will now transform the dataset into a binary format. Notice the changed attribute names:

#Attributes to be binarized:
binAttr = ['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Generation', 'Pr_Male', 'Height_m', 'Weight_kg', 'Catch_Rate']
#notBinAttr = ['isLegendary', 'hasGender', 'hasMegaEvolution']
notBinAttr = ['isLegendary', 'hasMegaEvolution']
#notBinAttr=list(dLogReg)

for col in binAttr:
    dLogReg = dLogReg.drop(col,axis=1)
    
#Removal of None and hasGender Column
dLogReg = dLogReg.drop('None',axis=1)
dLogReg = dLogReg.drop('hasGender',axis=1)
Xb=np.asarray(dLogTransform[binAttr])

negList=[]
for col in notBinAttr:
    negList.append("Inverse - "+col)

Xbin, attributeNamesBin = binarize2(Xb, binAttr)
print("X, i.e. the wine dataset, has now been transformed into:")
print(Xbin)
print(attributeNamesBin)

Xneg=1-np.asarray(dLogReg[notBinAttr])
X=np.concatenate((Xbin,np.asarray(dLogReg),Xneg),axis=1)
attributeNames=attributeNamesBin + list(dLogReg) + negList


T = mat2transactions(X,labels=attributeNames)
rules = apriori(T, min_support=.1, min_confidence=.9)
print_apriori_rules(rules)