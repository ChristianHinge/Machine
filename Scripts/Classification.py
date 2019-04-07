from ProjectData import *
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, tree
from toolbox_02450 import rocplot, confmatplot
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from matplotlib.image import imread
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show
import graphviz as gv

#Best lambda fra LogReg
lambda_interval = np.logspace(-8, 2, 10)

#Best Depth
tc = np.arange(2, 13, 1)
criterion='gini'

target='isLegendary'

# Class indices
y = np.array(dOriginal[target])
dOriginal=dOriginal.drop(target,axis=1)
dNorm=dNorm.drop(target,axis=1)


# Attribute names
attributeNames = list(dNorm)

# Attribute values
X = np.asarray(np.array(dNorm))

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)
#CV = model_selection.StratifiedKFold(n_splits=K)

# Initialize variables
Error_logreg = np.empty((K,1))
Opt_lambda = np.empty((K,1))
Error_dectree = np.empty((K,1))
Opt_depth = np.empty((K,1))
n_tested=0

k=0
for train_index, test_index in CV.split(X,y):
    print('CV-fold {0} of {1}'.format(k+1,K))
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit and evaluate Logistic Regression classifier
    Error_logreg_inner = np.empty((10,1))
    Error_dectree_inner = np.empty((10,1))
    for i in range(10):
        model = LogisticRegression(penalty='l2', C=1/lambda_interval[i], solver='liblinear' )
        model = model.fit(X_train, y_train)
        y_logreg = model.predict(X_test)
        Error_logreg_inner[i] = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)
        
        # Fit and evaluate Decision Tree classifier
        model2 = tree.DecisionTreeClassifier(criterion=criterion, max_depth=tc[i])
        model2 = model2.fit(X_train, y_train)
        y_dectree = model2.predict(X_test)
        Error_dectree_inner[i] = 100*(y_dectree!=y_test).sum().astype(float)/len(y_test)

    Error_dectree[k]=np.min(Error_dectree_inner)
    Opt_depth[k]=tc[np.argmin(Error_dectree_inner)]
    Error_logreg[k]=np.min(Error_logreg_inner)
    Opt_lambda[k]=lambda_interval[np.argmin(Error_logreg_inner)]
    k+=1
# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
z = (Error_logreg-Error_dectree)
zb = z.mean()
nu = K-1
sig =  (z-zb).std()  / np.sqrt(K-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
    
# Boxplot to compare classifier error distributions
figure()
boxplot(np.concatenate((Error_logreg, Error_dectree),axis=1))
xlabel('Logistic Regression   vs.   Classification Tree')
ylabel('Cross-validation error [%]')

show()

print('Ran Exercise 6.3.1')