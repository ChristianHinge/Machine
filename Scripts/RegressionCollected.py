import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net,rlr_validate
from scipy import stats
from tabulate import tabulate
from ProjectData import * #All of our data


# Data preparation and class attribute selection
sd_atk = np.std(dOriginal["Attack"])
mu_atk = np.mean(dOriginal["Attack"])
ChosenAttributes = ["HP","Defense","Sp_Atk","Sp_Def","Speed","isLegendary","Weight_kg"]
y_att = "Attack"
y = dNorm[y_att]
X = dNorm.drop(y_att,axis=1) #dNorm is already normalized

#X = X[ChosenAttributes]
attributeNames = list(X)
y = np.matrix(np.array(y)).transpose()
X = np.array(X)
N, M = X.shape

#### Cross validation parameters

K2 = 10 #Outer K-fold
K1 = 10  #Innter K-fold

CV_inner = model_selection.KFold(K1, shuffle=True)
CV_outer = model_selection.KFold(K2,shuffle=True)

#### Parameters for neural network classifier

iterations = np.array(range(1,12))*100 #The inner cross-validation fold will find the best parameters among these
opt_n_iterations = [] # The best parameters found in the inner CV folds, to be used in the outer folds

n_replicates = 3
loss_fn = torch.nn.MSELoss()

learning_curves = [] # for plotting

# Variables for storing the best net for later plotting
mse_best_ANN = 100 
y_est_best_ANN = None
y_true_best_ANN = None
net_best_hidden_units = None
best_net = None

def model():
    return lambda: torch.nn.Sequential(
        torch.nn.Linear(M, 2), #M features to n_iteration
        torch.nn.Tanh(),   # 1st transfer function,
        torch.nn.Linear(2, 1), # n_iteration to 1 output neuron
        # no final tranfer function, i.e. "linear output"
        )

def trainANN(max_iterations,n_rep):
    return train_neural_net(model(),
                            loss_fn,
                            X=X_train,
                            y=y_train,
                            n_replicates=n_rep,
                            max_iter=max_iterations)


#### Parameters for linear regression

lambdas = np.array(list(range(21)),dtype=int)*10 # The inner cross-validation folds will find the best parameters among these
lambdas[0] = 1
print(lambdas)
opt_lambdas = [] # The best parameters found in the inner CV folds, to be used in the outer folds

# Variables for storing the best net for later plotting
mse_best_LM = 100
y_est_best_LM = None
y_true_best_LM = None
lm_best_lambda = None

#Custom functiom which makes a linear model from X_train and y_train with regularization lambda
#and returns the prediction based on X_test. It assumes that the data is already normalized.
def reg_lm(lambda_,X_train,y_train,X_test):

    X = X_train
    X = np.concatenate((np.ones((X.shape[0],1)),X),1)
    mu = np.mean(X_train[:, 1:], 0)
    X_train[:, 1:] = (X_train[:, 1:] - mu ) 
    X_test[:, 1:] = (X_test[:, 1:] - mu ) 
    
    # Matrices for use in normal equation
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = lambda_ * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term

    # Solve the normal equation
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    y_train_est = X_train @ w_rlr.T
    y_test_est = X_test @ w_rlr.T

    return y_train_est.squeeze().transpose(), y_test_est.squeeze().transpose()


### ERRORS ####

#Errors with scaling. Each element will be scaled by |y_test_outer|. This will almost always be 1/10, but not exactly.
#The sum of these will yield the overall MSE for ANN, RLM and the Baseline
errors_ANN_outer = np.zeros(K2)
errors_baseline_outer = np.zeros(K2)
errors_LM_outer = np.zeros(K2)

#Errors without observation-size-fraction scaling. These are for plotting and the table
errors_ANN_outer_ns = np.zeros(K2)
errors_baseline_outer_ns = np.zeros(K2)
errors_LM_outer_ns = np.zeros(K2)

#Outer CV
for(k2,(train_index_o,test_index_o)) in enumerate(CV_outer.split(X,y)):
    
    #MSE error-matrices for inner cross validation
    #There's no paramater to tune for the baseline
    errors_ANN_inner = np.zeros((K1,len(iterations)))
    errors_LM_inner = np.zeros((K1,len(lambdas)))

    #Number of observations used for training and testing in inner CV
    #Equivelant to |D_par_i| in algorithm 6. Should be roughly 9/10*N
    N_inner = len(train_index_o)

    #Inner CV
    for (k1, (train_index, test_index)) in enumerate(CV_inner.split(X[train_index_o],y[train_index_o])):

        print('\nCrossvalidition\n\tOuter fold: {0}/{1}\n\tInner fold: {2}/{3} '.format(k2+1,K2,k1+1,K1))
        
        # ANN: Test each n_iteration.
        for iteration_index,n_iteration in enumerate(iterations):

            # Extract training and test set for current CV fold, convert to tensor
            X_train = torch.tensor(X[train_index,:], dtype=torch.float)
            y_train = torch.tensor(y[train_index], dtype=torch.float)
            X_test = torch.tensor(X[test_index,:], dtype=torch.float)
            y_test = torch.tensor(y[test_index], dtype=torch.float)
            
            # Train the net on training data
            net, final_loss, learning_curve = trainANN(n_iteration,n_replicates)
            
            # Determine estimated class labels for test set
            y_test_est = net(X_test)
            
            # Determine errors
            se = np.square(np.array(list(y_test_est.float()-y_test.float()))) # squared error
            mse = sum(se)/len(se) #mean
            errors_ANN_inner[k1,iteration_index] = mse*len(test_index)/N_inner # Algorithm 6
        

        # RLM: Test each lambda.
        for l_index,l in enumerate(lambdas):
            # Model and predict
            y_train_est, y_test_est = reg_lm(l, X[train_index],y[train_index],X[test_index])

            # Determine the errors
            se = np.power((y_test_est-y[test_index]),2)
            mse = sum(se)/len(y_test_est)
            errors_LM_inner[k1,l_index] = mse*len(test_index)/N_inner
    
    #### ANN OUTER: Train on newly found best parameter
    opt_hidden_unit = iterations[np.argmin(np.sum(errors_ANN_inner,axis=0))]
    opt_n_iterations.append(opt_hidden_unit)

    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.tensor(X[train_index_o,:], dtype=torch.float)
    y_train = torch.tensor(y[train_index_o], dtype=torch.float)
    X_test = torch.tensor(X[test_index_o,:], dtype=torch.float)
    y_test = torch.tensor(y[test_index_o], dtype=torch.float)

    # Train the net and predict Attack power
    net, final_loss, learning_curve = trainANN(opt_hidden_unit,n_replicates)
    y_test_est = net(X_test)
    learning_curves.append(learning_curve)
    
    # Determine MSE
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors_ANN_outer[k2] = mse*len(se)/N
    errors_ANN_outer_ns[k2] = mse

    #Save the best net for plotting.
    if mse < mse_best_ANN:
        y_est_best_ANN = y_test_est.data.numpy() 
        y_true_best_ANN = y_test.data.numpy()
        mse_best_ANN = mse
        net_best_hidden_units = opt_hidden_unit
        best_net = net #Save for plotting

    #### RLM: Train on newly found best lambda
    argmin = np.argmin(np.sum(errors_LM_inner,axis=0))
    opt_lambda = lambdas[argmin]
    opt_lambdas.append(opt_lambda)

    # Train and predict attack power
    y_train_est, y_test_est = reg_lm(opt_lambda, X[train_index_o],y[train_index_o],X[test_index_o])

    # Determine MSE
    se = np.square(y_test_est-y[test_index_o]) # squared error
    mse = np.squeeze(sum(se)/len(y_test_est)) #mean
    errors_LM_outer[k2] = mse*len(se)/N #Algorithm 6
    errors_LM_outer_ns[k2] = mse

    # Save the best RLM for plotting
    if mse < mse_best_LM:
        y_est_best_LM = y_test_est
        y_true_best_LM = y[test_index_o]
        mse_best_LM = mse
        lm_best_lambda = opt_lambda

    #### BASELINE
    se = np.square(y[test_index_o]-np.mean(y[train_index_o]))
    mse = sum(se)/len(se)
    errors_baseline_outer[k2] = mse*len(se)/N #Algorithm 6
    errors_baseline_outer_ns[k2] = mse


#### Plotting Results

# ANN Plotting

# Neural Net
weights = [best_net[i].weight.data.numpy().T for i in [0,2]]
biases = [best_net[i].bias.data.numpy() for i in [0,2]]
tf =  ["" for i in [1,2]]
#draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Learning curve and MSE
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
            'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

for i,learning_curve in enumerate(learning_curves):
    h, = summaries_axes[0].plot(learning_curve, color=color_list[i])
    h.set_label('CV fold {0}'.format(i+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
    #Display the MSE across folds
    summaries_axes[1].bar(np.arange(1, K2+1), errors_ANN_outer_ns, color=color_list)
    summaries_axes[1].set_xlabel('Outer fold')
    summaries_axes[1].set_xticks(np.arange(1, K2+1))
    summaries_axes[1].set_ylabel('MSE')
    summaries_axes[1].set_title('Test mean-squared-error')

summaries_axes[0].set_xlim([0,max(opt_n_iterations)])
summaries_axes[0].legend(["Outer fold {}: w. {} iterations".format(a+1,b) for a , b in enumerate(opt_n_iterations)])
plt.savefig('../Figures/ANN_training.png')

# Best net - True values vs. predicted
plt.figure(figsize=(6,6))
y_est = y_est_best_ANN*sd_atk+mu_atk
y_true = y_true_best_ANN*sd_atk+mu_atk
axis_range = [np.min([y_est, y_true])-20,np.max([y_est, y_true])+20]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Best net using {} iterations. MSE: {number:.{digits}f}'.format(net_best_hidden_units,number=mse_best_ANN[0],digits=3)])
plt.title('ANN predicting Attack power')
plt.ylim(axis_range) 
plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()
plt.savefig('../Figures/ANN_best_net.png')

# RLM Plotting

#Best RLM - Estimated vs. True
plt.figure(figsize=(6,6))
y_est = y_est_best_LM*sd_atk+mu_atk 
y_true = y_true_best_LM*sd_atk+mu_atk

axis_range = [np.min([y_est, y_true])-20,np.max([y_est, y_true])+20]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Best RLM using lambda = {}. MSE: {number:.{digits}f}'.format(lm_best_lambda,number=float(mse_best_LM),digits=3)])
plt.title('Regularized linear model predicting Attack power')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()
plt.savefig('../Figures/RLM_best_net.png')


#Latex table for regressoin B
data_table = np.zeros((K2,6))
data_table[:,0] = np.array(range(K2),dtype=int) + 1
data_table[:,1] = opt_hidden_unit
data_table[:,2] = errors_ANN_outer_ns
data_table[:,3] = opt_lambdas
data_table[:,4] = errors_LM_outer_ns
data_table[:,5] = errors_baseline_outer_ns

with open("../Figures/regression_b_tex.txt",'w') as handle:
    handle.writelines(tabulate(data_table, tablefmt="latex", floatfmt=".3f"))

###Final MSE for the three models and optimal parameters
print("Optimal RLM lambdas: {}".format(opt_lambdas))
print("Optimal ANN max iterations: {}".format(opt_n_iterations))

print('ANN: Estimated generalization error, MSE: {0}'.format(round(np.sum(errors_ANN_outer), 4)))
print('Baseline: Estimated generalization error, MSE: {0}'.format(round(np.sum(errors_baseline_outer), 4)))
print('Linear Regression: Estimated generalization error, MSE: {0}'.format(round(np.sum(errors_LM_outer), 4)))



def ConfidenceInterval(x,confidence=.95):
    n = len(x)
    mu = np.mean(x)
    std_err = np.std(x)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return [mu-h,mu+h]


print("\n=== Paired T-tests and confidence intervals ===\n")
print("LM vs Baseline")
print(stats.ttest_rel(errors_LM_outer_ns,errors_baseline_outer_ns))
print(ConfidenceInterval(errors_LM_outer_ns-errors_baseline_outer_ns))
print("ANN vs Baseline")
print(stats.ttest_rel(errors_ANN_outer_ns,errors_baseline_outer_ns))
print(ConfidenceInterval(errors_ANN_outer_ns-errors_baseline_outer_ns))
print("ANN vs LM")
print(stats.ttest_rel(errors_ANN_outer_ns,errors_LM_outer_ns))
print(ConfidenceInterval(errors_ANN_outer_ns-errors_LM_outer_ns))