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
#from tabulate import tabulate
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
X = np.array(dNorm)
N, M=X.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K2 = 10 #Outer K-fold
K1 = 10  #Innter K-fold

CV_inner = model_selection.KFold(K1, shuffle=True)
CV_outer = model_selection.KFold(K2,shuffle=True)



 


#Best lambda and depth values will be saved here
opt_lambdas = []
opt_depths = []


### ERRORS ####

# Variables for storing the best net for later plotting
#LogReg
error_best_LR = 100
y_est_best_LR = None
y_true_best_LR = None
LR_best_lambda = None

#Classification Tree
error_best_CT = 100 
y_est_best_CT = None
y_true_best_CT = None
net_best_depth = None
best_depth = None


#Errors with scaling. Each element will be scaled by |y_test_outer|. This will almost always be 1/10, but not exactly.
#The sum of these will yield the overall MSE for ANN, RLM and the Baseline
errors_CT_outer = np.zeros(K2)
errors_baseline_outer = np.zeros(K2)
errors_LogReg_outer = np.zeros(K2)

#Errors without observation-size-fraction scaling. These are for plotting and the table
errors_CT_outer_ns = np.zeros(K2)
errors_baseline_outer_ns = np.zeros(K2)
errors_LogReg_outer_ns = np.zeros(K2)

k_o=0
for train_index_o, test_index_o in CV_outer.split(X,y):
    k_in=0
    
    #Hit error-matrices for inner cross validation
    #There's no paramater to tune for the baseline
    errors_CT_inner = np.zeros((K1,len(tc)))
    errors_LogReg_inner = np.zeros((K1,len(lambda_interval)))
    baseline = np.zeros(y[test_index_o].shape) 

    #Number of observations used for training and testing in inner CV
    #Equivelant to |D_par_i| in algorithm 6. Should be roughly 9/10*N
    N_inner = len(train_index_o)
    
    
    for train_index, test_index in CV_inner.split(X[train_index_o],y[train_index_o]):
        print('\nCrossvalidition\n\tOuter fold: {0}/{1}\n\tInner fold: {2}/{3} '.format(k_o,K2,k_in+1,K1))
        
        # extract training and test set for current CV fold
        X_train = X[train_index,:]
        y_train = y[train_index]
        X_test = X[test_index,:]
        y_test = y[test_index]
    
        #Train Logistic Regression classifier
        for lambda_index, n_lambda in enumerate(lambda_interval):
            model = LogisticRegression(penalty='l2', C=1/lambda_interval[lambda_index], solver='liblinear' )
            model = model.fit(X_train, y_train)
            y_logreg = model.predict(X_test)
            
            #Calculate Hit rate Error
            error = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)
            errors_LogReg_inner[k_in, lambda_index]=error*len(test_index)/N_inner
           
        #Train Decision Tree classifier    
        for depth_index, depth in enumerate(tc):
            model2 = tree.DecisionTreeClassifier(criterion=criterion, max_depth=tc[depth_index])
            model2 = model2.fit(X_train, y_train)
            y_dectree = model2.predict(X_test)
            
            #Calculate Hit rate Error    
            error = 100*(y_dectree!=y_test).sum().astype(float)/len(y_test)
            errors_CT_inner[k_in, depth_index]=error*len(test_index)/N_inner
            
        k_in+=1    
    
    ####Outer loop####
    #Save optimal parameters
    opt_lambda = lambda_interval[np.argmin(np.sum(errors_LogReg_inner,axis=0))]
    opt_lambdas.append(opt_lambda)
    
    opt_depth = tc[np.argmin(np.sum(errors_CT_inner,axis=0))]
    opt_depths.append(opt_depth)
    
    
    # extract training and test set for current CV fold
    X_train = X[train_index_o,:]
    y_train = y[train_index_o]
    X_test = X[test_index_o,:]
    y_test = y[test_index_o]
    
    #Train  Logistic Regression classifier again
    model = LogisticRegression(penalty='l2', C=1/opt_lambda, solver='liblinear' )
    model = model.fit(X_train, y_train)
    y_logreg = model.predict(X_test)
    
    #Calculate Hit rate Error
    error = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)
    errors_LogReg_outer[k_o]=error*len(test_index_o)/N_inner
    errors_LogReg_outer_ns[k_o]=error
    
    #Save best lambda
    if error < error_best_LR:
        y_est_best_LR = y_test
        y_true_best_LR = y[test_index_o]
        error_best_LR = error
        LR_best_lambda = opt_lambda
    
    
    #Train Decision Tree classifier  
    model2 = tree.DecisionTreeClassifier(criterion=criterion, max_depth=opt_depth)
    model2 = model2.fit(X_train, y_train)
    y_dectree = model2.predict(X_test)
    
    #Calculate Hit rate Error
    error = 100*(y_dectree!=y_test).sum().astype(float)/len(y_test)
    errors_CT_outer[k_o]=error*len(test_index_o)/N_inner   
    errors_CT_outer_ns[k_o]=error
    
    #Save best depth
    if error < error_best_CT:
        y_est_best_CT = y_test
        y_true_best_CT = y_test
        bestTree = model2
        error_best_CT = error
        best_depth = opt_depth
        
        
    error = 100*(baseline!=y_test).sum().astype(float)/len(y_test)
    errors_baseline_outer[k_o] = error*len(test_index_o)/N 
    errors_baseline_outer_ns[k_o] = error     
    
    k_o+=1
# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
z = (errors_LogReg_outer-errors_CT_outer)
zb = z.mean()
nu = K2-1
sig =  (z-zb).std()  / np.sqrt(K2-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

#
print("The confidence interval is[{},{}])".format(zL,zH))

"""
if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
"""
RapportMatrix=np.hstack((np.array(opt_depths).reshape(-1,1),errors_CT_outer.reshape(-1,1), np.array(opt_lambdas).reshape(-1,1),errors_LogReg_outer.reshape(-1,1), errors_baseline_outer.reshape(-1,1)))    
   
ErrorMatrix=np.hstack((errors_LogReg_outer.reshape(-1,1), errors_CT_outer.reshape(-1,1), errors_baseline_outer.reshape(-1,1)))    
# Boxplot to compare Classification tree with Logistic Regression
figure()
boxplot(np.concatenate((errors_LogReg_outer.reshape(-1,1), errors_CT_outer.reshape(-1,1)),axis=1))
xlabel('Logistic Regression   vs.   Classification Tree')
ylabel('Cross-validation error [%]')
show()

# Boxplot to compare Classification tree with Logistic Regression
figure()
boxplot(ErrorMatrix)
xlabel('Logistic Regression   vs.   Classification Tree    vs. Baseline')
ylabel('Cross-validation error [%]')
show()

out = tree.export_graphviz(bestTree, out_file='tree_Legendary.dot', feature_names=attributeNames)
#Kør kommandoen "dot -Tpng tree_Legendary.dot -o tree_Legendary.png"
# fra working directory for at se plot

print(RapportMatrix)