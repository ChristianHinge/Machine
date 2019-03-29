from CarsData import *
from matplotlib import pyplot as plt
import numpy as np

#Dataframe with original non-nominal attributes
dForMatrix = dOriginal[attNoK]

#attsToBeRemoved=["num-of-doors","num-of-cylinders","symboling"]
#for col in attsToBeRemoved:
#    attNoK.remove(col)

#Dataframe with normalized continouus attributes
dForCorr = dNorm[attNoK]

######### Standard Statistics #############    
statMatrix=pd.DataFrame(data=np.zeros((12,3)),index=list(dForMatrix), columns=["Mean","Variance", "Standard Deviation"])
for col in list(dForMatrix):
     statMatrix["Mean"][col]=round(np.mean(dForMatrix[col].values),2)
     statMatrix["Variance"][col]=round(np.var(dForMatrix[col].values),2)
     statMatrix["Standard Deviation"][col]=round(np.std(dForMatrix[col].values),2)
print(statMatrix)

#########Correlation Matrix#############
corMatrix=pd.DataFrame(data=np.corrcoef(dForCorr, rowvar=False),index=attNoK, columns=attNoK)

#Plotting the heatmap##

#Colormap
cmap = plt.get_cmap('PuOr')
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

fig, ax = plt.subplots()
im = ax.imshow(corMatrix.values,interpolation='none',cmap=cmap)

#Axis labels
ax.set_xticks(np.arange(len(list(corMatrix))))
ax.set_yticks(np.arange(len(list(corMatrix))))
ax.set_xticklabels(list(corMatrix))
ax.set_yticklabels(list(corMatrix))

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation='vertical')

"""
# Loop over data dimensions and create text annotations.
for i in range(len(list(corMatrix))):
    for j in range(len(list(corMatrix))):
        text = ax.text(j, i, round(corMatrix.values[i, j],2),
                       ha="center", va="center", color="w")
"""
ax.set_title("Correlation matrix of continous attributes")
fig.colorbar(im)
fig.tight_layout()
plt.savefig("../Figures/CorrelationMatrix")
plt.show()


######### Scatter-Plot ############

M=len(list(dForCorr))

#Plotting in columns of 5 and rows of 14
#3 Figures in total
lowerBound = [0,4,8]
upperBound = [4,8,12]

for l,u in zip(lowerBound,upperBound):

    #Figure, size, and title
    fig, axes = plt.subplots(nrows=12, ncols=u-l)
    fig.suptitle("Scatterplots for: " + ', '.join(list(dForCorr)[l:u]),fontsize = 16)
    fig.set_size_inches(14*0.7,23*0.7)
    ratio = fig.get_size_inches()[0]/fig.get_size_inches()[1]

    #The column number in the figure.
    plotCol = 0
    
    for m1 in range(l,u):

        for m2 in range(M):

            #Plot values         
            axes[m2][plotCol].plot(dForCorr.values[:,m2], dForCorr.values[:,m1], '.')

            # Hide labels for subplots in the middle
            axes[m2][plotCol].set_xticks([])
            axes[m2][plotCol].set_yticks([])

            # If the subplot is at the left or bottom edge, show labels
            if m2 == 11:
                axes[m2][plotCol].set_xlabel(list(dForCorr)[m1])
            if m1 == l:
                axes[m2][plotCol].set_ylabel(list(dForCorr)[m2])
            
        plotCol = plotCol + 1
    
    #Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(wspace=0.1*ratio, hspace=0.1)
    plt.savefig("../Figures/ScatterPlots/ScatterPlot"+str(l+1)+"-"+str(u)+".png",dpi=300)    
