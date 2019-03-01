import pandas as pd
import pickle

dOriginal = pd.read_pickle("dOriginal")
dNorm = pd.read_pickle("dNorm")
with open('1_hot_K_dict.pickle', 'rb') as handle:
    OneHotKDict = pickle.load(handle)
