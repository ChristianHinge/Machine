from CarsData import *
from matplotlib import pyplot as plt
import numpy as np

plt.figure(figsize=(14,9))

#Sqrt ensures that there are as many rows as there are cols of subplots
u = np.floor(np.sqrt(len(attNoK))); v = np.ceil(float(len(attNoK))/u)

i = 1
for col in attNoK:
    plt.subplot(u,v,i)
    plt.hist(dOriginal[col])
    plt.xlabel(col)
    i = i+1

plt.suptitle('Histogram of non-nominal attributes',fontsize=18)
plt.tight_layout()
plt.gcf().subplots_adjust(top=0.93)
plt.savefig('../Figures/AttributeHistograms.png')
