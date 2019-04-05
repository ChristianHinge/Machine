import pandas as pd
import pickle

#Load data

dOriginal = pd.read_pickle("../Data/dOriginal")
dNorm = pd.read_pickle("../Data/dNorm")
dLogTransform = pd.read_pickle("../Data/dLogTransform")
dLogReg = pd.read_pickle("../Data/dLogReg")

with open('../Data/1_hot_K_dict.pickle', 'rb') as handle:
    oneHotKDict = pickle.load(handle)
with open('../Data/non_hot_k_list.pickle', 'rb') as handle:
    attNoK = pickle.load(handle)   
