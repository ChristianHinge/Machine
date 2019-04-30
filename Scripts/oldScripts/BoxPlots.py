from matplotlib import pyplot as plt
from ProjectData import *

for col in attNoK:
    plt.figure()
    plt.boxplot(dOriginal[col].values)
    plt.ylabel(col,fontsize=13)
    plt.xticks([])
    #plt.title(col+' - boxplot')
    plt.savefig("../Figures/BoxPlots/"+col+".png")