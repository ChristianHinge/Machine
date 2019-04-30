# exercise 11.1.5
from matplotlib.pyplot import figure, plot, legend, xlabel, show
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from sklearn import model_selection
from ProjectData import *
from matplotlib.pyplot import cm
from toolbox_02450 import clusterval

# Load Matlab data file and extract variables of interest
selected_K = 4
y = np.array(dNorm["isLegendary"],dtype=int)
dNorm = dNorm.drop("isLegendary",axis=1)
attributeNames = list(dNorm)
X = np.array(dNorm)
classNames = ["Not legendary","Legendary"]
N, M = X.shape
C = len(classNames)

centroid_data = np.zeros((dNorm.shape[0],selected_K))
# Range of K's to try
KRange = [4]
T = len(KRange)

covar_type = 'full'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        if K == selected_K:
            for mean_i,mean in enumerate(gmm.means_):
                print("Centroid {}".format(mean_i+1))
                i = 0
                for lab, val in zip(attributeNames,mean):
                    #if lab in list(dOriginal):
                    val = val/np.std(dNorm[lab])
                    centroid_data[i,mean_i] = val
                    #print("{}:{}".format(lab,round(val,2)))
                    i += 1
                print("\n\n")
        print(gmm.means_.shape)
        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results
"""
figure(1); 
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
#plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC'])
xlabel('K')
show()

"""

b = gmm.predict(X)
Rand, Jaccard, NMI = clusterval(b,y)   
print(Rand)
print(Jaccard)
print(NMI)



intervals = list(range(0,dNorm.shape[1],23))+[dNorm.shape[1]]

figs, axs = plt.subplots(nrows = 1, ncols = 3)
figs.set_figheight(15)
figs.set_figwidth(15)
for i in range(len(intervals)-1):
    lower = intervals[i]
    upper = intervals[i+1]
    print(type(axs))
    
    #col = i%3
    ax = axs[i]
    ax.matshow(centroid_data[lower:upper])
    ax.set_xticklabels(["","1","2","3","4"])
    ax.set_yticks(range(23))
    ax.set_yticklabels(attributeNames[lower:upper])

figs.suptitle("GMM Clustering",fontsize=30)
#figs.tight_layout()
plt.savefig("../Figures/Centroids.png",transparent=True)
print('Ran Exercise 11.1.5')

