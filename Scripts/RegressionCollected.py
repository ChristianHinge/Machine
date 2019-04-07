#region imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
import sklearn.linear_model as lm
from toolbox_02450 import train_neural_net, draw_neural_net,rlr_validate
from scipy import stats
from ProjectData import *
#endregion

#region Data preparation and class attribute selection
sd_atk = np.std(dOriginal["Attack"])
mu_atk = np.mean(dOriginal["Attack"])
ChosenAttributes = ["HP","Defense","Sp_Atk","Sp_Def","Speed","isLegendary","Weight_kg"]
y_att = "Attack"
y = dNorm[y_att]
X = dNorm.drop(y_att,axis=1)

#X = X[ChosenAttributes]
attributeNames = list(X)
y = np.matrix(np.array(y)).transpose()
X = np.array(X)
N, M = X.shape

#endregion

#region Cross validation parameters
K2 = 10 #Outer K-fold
K1 = 10  #Innter K-fold

CV_inner = model_selection.KFold(K1, shuffle=True)
CV_outer = model_selection.KFold(K2,shuffle=True)
#endregion


#region Parameters for neural network classifier
doANN = True
hidden_unit_start = 1 #inclusive
hidden_unit_end   = 6  #exclusive    
neurons = np.array(list(range(hidden_unit_start,hidden_unit_end)))*200

n_replicates = 1
#max_iter = 1500
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
errors_ANN_outer = np.zeros(K2)
errors_ANN_outer_ns = np.zeros(K2)
learning_curves = []
optimal_n_hidden_units = []

mse_best_ANN = 100
y_est_best_ANN = None
y_true_best_ANN = None
net_best_hidden_units = None
best_net = None

def model():
    return lambda: torch.nn.Sequential(
        torch.nn.Linear(M, 2), #M features to n_hidden_units
        torch.nn.Tanh(),   # 1st transfer function,
        torch.nn.Linear(2, 1), # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
        )
def trainANN(max_iterations,n_rep):
    return train_neural_net(model(),
                                        loss_fn,
                                        X=X_train,
                                        y=y_train,
                                        n_replicates=n_rep,
                                        max_iter=max_iterations)
#endregion

#### Parameters for linear regression

lambdas = list(np.power(10.,range(-5,9)))
#lambdas = [1/100000,1/10000,1/1000,1/100,1/10,1,10,100,1000,10000,100000,1000000]
#lambdas = [10,100]
opt_lambdas = []
mse_best_LM = 100
y_est_best_LM = None
y_true_best_LM = None
lm_best_lambda = None
errors_LM_outer = np.zeros(K2)
errors_LM_outer_ns = np.zeros(K2)

def reg_lm(lambda_,X_train,y_train,X_test,y_test):

    X = X_train
    X = np.concatenate((np.ones((X.shape[0],1)),X),1)
    mu = np.mean(X_train[:, 1:], 0)
    #The following not work due to random splits
    """
    sigma = np.std(X_train[:, 1:], 0)
    sigma[np.isnan(sigma)] = 1
    sigma[sigma==0] = 1
    """
    X_train[:, 1:] = (X_train[:, 1:] - mu ) 
    X_test[:, 1:] = (X_test[:, 1:] - mu ) 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = lambda_ * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    y_train_est = X_train @ w_rlr.T
    y_test_est = X_test @ w_rlr.T
    return y_train_est.squeeze().transpose(), y_test_est.squeeze().transpose()

errors_baseline = np.zeros(K2)
errors_baseline_ns = np.zeros(K2)

for(k2,(train_index_o,test_index_o)) in enumerate(CV_outer.split(X,y)):
    
    #ANN
    errors_ANN_inner = np.zeros((K1,len(neurons)))
    errors_LM_inner = np.zeros((K1,len(lambdas)))
    se = np.square(y[test_index_o]-np.mean(y[train_index_o]))
    mse = sum(se)/len(se)
    errors_baseline[k2] = mse*len(se)/N
    errors_baseline_ns[k2] = mse

    for (k1, (train_index, test_index)) in enumerate(CV_inner.split(X[train_index_o],y[train_index_o])):

        print('\nCrossvalidition\n\tOuter fold: {0}/{1}\n\tInner fold: {2}/{3} '.format(k2,K2,k1+1,K1))
        
        #region ANN testing for each hidden neuron configuration
        if doANN:
        
            for hidden_index,n_hidden_units in enumerate(neurons):

                print('Max iterations = {}'.format(n_hidden_units))
                # Extract training and test set for current CV fold, convert to tensor
                X_train = torch.tensor(X[train_index,:], dtype=torch.float)
                y_train = torch.tensor(y[train_index], dtype=torch.float)
                X_test = torch.tensor(X[test_index,:], dtype=torch.float)
                y_test = torch.tensor(y[test_index], dtype=torch.float)
                
                # Train the net on training data
                net, final_loss, learning_curve = trainANN(n_hidden_units,2)
                print('\n\tBest loss: {}\n'.format(final_loss))
                
                # Determine estimated class labels for test set
                y_test_est = net(X_test)
                
                # Determine errors and errors TJEK OM NEDENSTÅENDE GIVER VEKTOR
                se = np.square(np.array(list(y_test_est.float()-y_test.float()))) # squared error
                mse = sum(se)/len(se) #mean
                errors_ANN_inner[k1,hidden_index] = mse
        #endregion


        for l_index,l in enumerate(lambdas):

            y_train_est, y_test_est = reg_lm(l, X[train_index],y[train_index],X[test_index],y[test_index])
            se = np.power((y_test_est-y[test_index]),2)
            mse = sum(se)/len(y_test_est)
            errors_LM_inner[k1,l_index] = mse
    
    argmin = np.argmin(np.mean(errors_LM_inner,axis=0))
    opt_lambda = lambdas[argmin]
    opt_lambdas.append(opt_lambda)

    y_train_est, y_test_est = reg_lm(opt_lambda, X[train_index_o],y[train_index_o],X[test_index_o],y[test_index_o])
    se = np.power(y_test_est-y[test_index_o],2) # squared error
    mse = (sum(se)/len(y_test_est)) #mean
    errors_LM_outer[k2] = mse*len(se)/N
    errors_LM_outer_ns[k2] = mse

    if mse < mse_best_LM:
        y_est_best_LM = y_test_est
        y_true_best_LM = y[test_index_o]
        mse_best_LM = mse
        lm_best_lambda = opt_lambda

    # region ANN: learning on optimal hidden unit
    if doANN:
        opt_hidden_unit = neurons[np.argmin(np.mean(errors_ANN_inner,axis=0))]
        optimal_n_hidden_units.append(opt_hidden_unit)
    
        #### Train on d_train_outer with the newly found best parameters

        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.tensor(X[train_index_o,:], dtype=torch.float)
        y_train = torch.tensor(y[train_index_o], dtype=torch.float)
        X_test = torch.tensor(X[test_index_o,:], dtype=torch.float)
        y_test = torch.tensor(y[test_index_o], dtype=torch.float)

        # Train the net and determine class labels
        net, final_loss, learning_curve = trainANN(opt_hidden_unit,4)
        y_test_est = net(X_test)
        learning_curves.append(learning_curve)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        errors_ANN_outer[k2] = mse*len(se)/N
        errors_ANN_outer_ns[k2] = mse

        #Save results if these are the best so far
        if mse < mse_best_ANN:
            y_est_best_ANN = y_test_est.data.numpy() 
            y_true_best_ANN = y_test.data.numpy()
            mse_best_ANN = mse
            net_best_hidden_units = opt_hidden_unit
            best_net = net #Save for plotting
    # endregion

if doANN:
    print('\nANN: Estimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.sum(errors_ANN_outer)), 4)))

print('\nBaseline: Estimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.sum(errors_baseline)), 4)))
print('\nLinear Regression: Estimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.sum(errors_LM_outer)), 4)))

##Plotting Data

plt.figure(figsize=(6,6))
y_est = y_est_best_LM*sd_atk+mu_atk 
y_true = y_true_best_LM*sd_atk+mu_atk

axis_range = [np.min([y_est, y_true])-20,np.max([y_est, y_true])+20]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Best RLM using lambda = {}'.format(lm_best_lambda)])
plt.title('Regularized linear model predicting Attack power')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()
plt.savefig('../Figures/RLM_best_net.png')
#plt.show()


#region ANN Plotting

print('Diagram of best neural net in last fold:')
weights = [best_net[i].weight.data.numpy().T for i in [0,2]]
biases = [best_net[i].bias.data.numpy() for i in [0,2]]
tf =  ["" for i in [1,2]]
#draw_neural_net(weights, biases, tf, attribute_names=attributeNames)


if doANN:    
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    #Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

    for i,learning_curve in enumerate(learning_curves):
        h, = summaries_axes[0].plot(learning_curve, color=color_list[i])
        h.set_label('CV fold {0}'.format(i+1))
        summaries_axes[0].set_xlabel('Iterations')
        #summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')
        

        #Display the MSE across folds
        summaries_axes[1].bar(np.arange(1, K2+1), errors_ANN_outer_ns, color=color_list)
        summaries_axes[1].set_xlabel('Outer fold')
        summaries_axes[1].set_xticks(np.arange(1, K2+1))
        summaries_axes[1].set_ylabel('MSE')
        summaries_axes[1].set_title('Test mean-squared-error')
    summaries_axes[0].set_xlim([0,max(optimal_n_hidden_units)])
    summaries_axes[0].legend(["Outer fold {}: w. {} iterations".format(a+1,b) for a , b in enumerate(optimal_n_hidden_units)])
    plt.savefig('../Figures/ANN_training.png')
    plt.figure(figsize=(6,6))

    y_est = y_est_best_ANN*sd_atk+mu_atk
    y_true = y_true_best_ANN*sd_atk+mu_atk

    axis_range = [np.min([y_est, y_true])-20,np.max([y_est, y_true])+20]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Best net using {} iterations'.format(net_best_hidden_units)])
    plt.title('ANN predicting Attack power')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()
    plt.savefig('../Figures/ANN_best_net.png')
    #plt.show()

print("Optimal lambdas: {}".format(opt_lambdas))
print("Optimal hidden neurons: {}".format(optimal_n_hidden_units))
with open ('../Figures/CrossValidaitonTable.txt', 'w') as handle:
    handle.writelines(["Optimal ANN iterations",str(optimal_n_hidden_units),"\n"])
    handle.writelines(["Generalization errors ANN",str(errors_ANN_outer_ns),"\n"])
    handle.writelines(["Optimal lambdas",str(opt_lambdas),"\n"])
    handle.writelines(["Generalization errors RLM",str(errors_LM_outer_ns),"\n"])
    handle.writelines(["Generalization errors Baseline",str(errors_baseline_ns),"\n"])


#endregion