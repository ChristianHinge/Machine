import pandas as pd
import pickle

dOriginal = pd.read_pickle("../Data/dOriginal")
dNorm = pd.read_pickle("../Data/dNorm")
with open('../Data/1_hot_K_dict.pickle', 'rb') as handle:
    OneHotKDict = pickle.load(handle)
