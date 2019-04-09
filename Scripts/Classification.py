from ProjectData import *
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection, tree
from toolbox_02450 import rocplot, confmatplot
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot, imshow
from matplotlib.image import imread
import matplotlib.pyplot as plt
from tabulate import tabulate
import graphviz as gv
from sklearn.metrics import confusion_matrix
#Best lambda fra LogReg
lambda_interval = np.logspace(-6, 2, 10)

#Best Depth
tc = np.arange(2, 13, 1)
criterion='gini'

target='isLegendary'

# Class indices
y = np.array(dOriginal[target],dtype=int)
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
LR_best_model = None
LR_saved_performance = np.zeros((K2,len(lambda_interval)))
LR_coefs = np.zeros((K2,len(attributeNames)))

#Classification Tree
error_best_CT = 100 
y_est_best_CT = None
y_true_best_CT = None
net_best_depth = None
best_depth = None
CT_saved_performance = np.zeros((K2,len(tc)))

#Baseline
error_best_BL = 100

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

    counts = np.bincount(y[train_index_o])
    baseline = np.argmax(counts)

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
    error_LogReg_inner_sum = np.sum(errors_LogReg_inner,axis=0)
    LR_saved_performance[k_o,:] = error_LogReg_inner_sum
    opt_lambda = lambda_interval[np.argmin(error_LogReg_inner_sum)]
    opt_lambdas.append(opt_lambda)

    error_CT_inner_sum = np.sum(errors_CT_inner,axis=0)
    CT_saved_performance[k_o,:] = error_CT_inner_sum
    opt_depth = tc[np.argmin(error_CT_inner_sum)]
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
    LR_coefs[k_o,:] = model.coef_
    
    #Calculate Hit rate Error
    error = 100*(y_logreg!=y_test).sum().astype(float)/len(y_test)
    errors_LogReg_outer[k_o]=error*len(test_index_o)/N_inner
    errors_LogReg_outer_ns[k_o]=error
    
    
    #Save best lambda
    if error < error_best_LR:
        y_est_best_LR = y_logreg
        y_true_best_LR = y_test
        error_best_LR = error
        LR_best_lambda = opt_lambda
        LR_best_model = model
    
    
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
        y_est_best_CT = y_dectree
        y_true_best_CT = y_test
        bestTree = model2
        error_best_CT = error
        best_depth = opt_depth
        
        
    error = 100*(baseline!=y_test).sum()/len(y_test)
    errors_baseline_outer[k_o] = error*len(test_index_o)/N 
    errors_baseline_outer_ns[k_o] = error     
    
    #Save best error for baseline
    if error < error_best_BL:
        y_est_best_BL = np.ones(len(y_test),dtype=int)*baseline
        y_true_best_BL = y_test
        error_best_BL = error
        best_BL = baseline    
    k_o+=1
# Test if classifiers are significantly different using methods in section 9.3.3
# by computing credibility interval. Notice this can also be accomplished by computing the p-value using
# [tstatistic, pvalue] = stats.ttest_ind(Error_logreg,Error_dectree)
# and test if the p-value is less than alpha=0.05. 
"""
z = (errors_LogReg_outer_ns-errors_CT_outer_ns)
zb = z.mean()
nu = K2-1
sig =  (z-zb).std()  / np.sqrt(K2-1)
alpha = 0.05

zL = zb + sig * stats.t.ppf(alpha/2, nu);
zH = zb + sig * stats.t.ppf(1-alpha/2, nu);

print("The confidence interval is[{},{}])".format(zL,zH))

if zL <= 0 and zH >= 0 :
    print('Classifiers are not significantly different')        
else:
    print('Classifiers are significantly different.')
"""


RapportMatrix=np.hstack((np.array(opt_depths).reshape(-1,1),errors_CT_outer.reshape(-1,1), np.array(opt_lambdas).reshape(-1,1),errors_LogReg_outer.reshape(-1,1), errors_baseline_outer.reshape(-1,1)))    
   
ErrorMatrix=np.hstack((errors_LogReg_outer.reshape(-1,1), errors_CT_outer.reshape(-1,1), errors_baseline_outer.reshape(-1,1)))  
"""  
# Boxplot to compare Classification tree with Logistic Regression
figure()
boxplot(np.concatenate((errors_LogReg_outer.reshape(-1,1), errors_CT_outer.reshape(-1,1)),axis=1))
xlabel('Logistic Regression   vs.   Classification Tree')
ylabel('Cross-validation error [%]')
#show()
"""
# Boxplot to compare Classification tree with Logistic Regression
figure()
boxplot(ErrorMatrix)
xlabel('Logistic Regression   vs.   Classification Tree    vs. Baseline')
ylabel('Cross-validation error [%]')
plt.grid()
plt.savefig("../Figures/comparison.png")


#Boxplot of weights
fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(10)
plt.title("Logistic Regression: Attribute weights")
plt.ylabel("Attribute weight")
plt.boxplot(LR_coefs)
plt.xticks(range(len(attributeNames)+1),[""]+attributeNames,rotation = 90)
plt.tight_layout()
plt.grid()
plt.savefig("../Figures/LR_weights.png",dpi=600)


#Parameter performance
y_dat = LR_saved_performance
y = np.mean(y_dat,axis=0)
yerr = np.abs(np.quantile(y_dat-y,[0.25,0.75],axis=0))
plt.figure(figsize=(6,6))
plt.errorbar(np.log10(lambda_interval), y, yerr=yerr, fmt='o-',capsize=5,capthick=3,ms=10)
plt.grid()
plt.xlabel("Log10(Lambda) values")
plt.ylabel("Average generalization error of lambda values")
plt.title('Logistic Regression: Performance of lambda parameter')
plt.savefig('../Figures/LR_lambda_performance.png')

y_dat = CT_saved_performance
y = np.mean(y_dat,axis=0)
yerr = np.abs(np.quantile(y_dat-y,[0.25,0.75],axis=0))
plt.figure(figsize=(6,6))
plt.errorbar(tc, y, yerr=yerr, fmt='o-',capsize=5,capthick=3,ms=10)
plt.grid()
plt.xlabel("Tree depth")
plt.ylabel('LogReg: Average generalization error of tree depth values')
plt.title("Decision tree: Performance of depth parameter")
plt.savefig('../Figures/LR_depth_performance.png')

out = tree.export_graphviz(bestTree, out_file='tree_Legendary.dot', feature_names=attributeNames)
#Kør kommandoen "dot -Tpng tree_Legendary.dot -o tree_Legendary.png"
# fra working directory for at se plot


#Confusion Matrix Classification Tree
cm1 = confusion_matrix(y_true_best_CT, y_est_best_CT)
accuracy = 100*cm1.diagonal().sum()/cm1.sum(); error_rate = 100-accuracy
figure()
plt.imshow(cm1, cmap='binary', interpolation='None')
plt.colorbar()
plt.xticks(range(2)); plt.yticks(range(2))
plt.xlabel('Predicted class'); plt.ylabel('Actual class')
plt.title('Classification Tree\n(Accuracy: {0}%, Error Rate: {1}%)'.format(np.round(accuracy,2), np.round(error_rate,2)));
plt.tight_layout()
plt.savefig("../Figures/ConfusionMatrix/ClassTree.png")

#Latex table for Classification
data_table = np.zeros((K2,6))
data_table[:,0] = np.array(range(K2),dtype=int) + 1
data_table[:,1] = opt_depths
data_table[:,2] = errors_CT_outer_ns
data_table[:,3] = opt_lambdas
data_table[:,4] = errors_LogReg_outer_ns
data_table[:,5] = errors_baseline_outer_ns

with open("../Figures/classification_tex.txt",'w') as handle:
    handle.writelines(tabulate(data_table, tablefmt="latex", floatfmt=".3f"))
    
def ConfidenceInterval(x,confidence=.95):
    n = len(x)
    mu = np.mean(x)
    std_err = np.std(x)
    h = std_err/np.sqrt(K2) * stats.t.ppf((1 - confidence) / 2, n - 1)
    return [mu-abs(h),mu+abs(h)]

print(LR_best_model.coef_)

###Final classification error for the three models and optimal parameters

print("\n\n")
print("Optimal LogReg lambdas: {}".format(opt_lambdas))
print("Optimal CT max iterations: {}".format(opt_depths))

print('LogReg: Estimated generalization error: {0}%'.format(round(np.sum(errors_LogReg_outer), 4)))
print('Baseline: Estimated generalization error: {0}%'.format(round(np.sum(errors_baseline_outer), 4)))
print('CT: Estimated generalization error: {0}%'.format(round(np.sum(errors_CT_outer), 4)))


print("\n=== Paired T-tests and confidence intervals ===\n")
print("CT vs Baseline")
print(stats.ttest_rel(errors_CT_outer_ns,errors_baseline_outer_ns))
print(ConfidenceInterval(errors_CT_outer_ns-errors_baseline_outer_ns))
print("LogReg vs Baseline")
print(stats.ttest_rel(errors_LogReg_outer_ns,errors_baseline_outer_ns))
print(ConfidenceInterval(errors_LogReg_outer_ns-errors_baseline_outer_ns))
print("LogReg vs CT")
print(stats.ttest_rel(errors_LogReg_outer_ns,errors_CT_outer_ns))
print(ConfidenceInterval(errors_LogReg_outer_ns-errors_CT_outer_ns))


#print(RapportMatrix)