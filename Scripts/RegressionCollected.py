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
#ChosenAttributes = ["HP","Defense","Sp_Atk","Sp_Def","Speed","isLegendary","Weight_kg"]
y_att = "Attack"
y = dNorm[y_att]
X = dNorm.drop(y_att,axis=1)
y = np.matrix(np.array(y)).transpose()
X = np.array(X)
attributeNames = list(X)
N, M = X.shape

# Normalize data
#X = stats.zscore(X);
#endregion

#region Cross validation parameters
K2 = 2 #Outer K-fold
K1 = 2 #Inner K-fold
CV_inner = model_selection.KFold(K1, shuffle=True)
CV_outer = model_selection.KFold(K2,shuffle=True)
#endregion

#region Parameters for neural network classifier
doANN = False
hidden_unit_start = 1 #inclusive
hidden_unit_end = 3  #exclusive    
n_replicates = 1
max_iter = 1000
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
errors_ANN_outer = np.zeros(K2)
learning_curves = []
optimal_n_hidden_units = []
mse_best_ANN = 100
y_est_best_ANN = None
y_true_best_ANN = None
net_best_hidden_units = None
def model(x):
    return lambda: torch.nn.Sequential(
        torch.nn.Linear(M, x), #M features to n_hidden_units
        torch.nn.Tanh(),   # 1st transfer function,
        torch.nn.Linear(x, 1), # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
        )
def trainANN(n_hidden):
    return train_neural_net(model(n_hidden),
                                        loss_fn,
                                        X=X_train,
                                        y=y_train,
                                        n_replicates=1,
                                        max_iter=max_iter)
#endregion

#### Parameters for baseline

#### Parameters for linear regression

for(k2,(train_index_o,test_index_o)) in enumerate(CV_outer.split(X,y)):
    print('\n=== Outer Crossvalidation fold: {0}/{1}'.format(k2+1,K2))
    
    #ANN
    errors_ANN_inner = np.zeros((K1,hidden_unit_end-hidden_unit_start))

    for (k1, (train_index, test_index)) in enumerate(CV_inner.split(X[train_index_o],y[train_index_o])):

        print('\nInner Crossvalidation fold: {0}/{1}'.format(k1+1,K1))
        
        #region ANN testing for each hidden neuron configuration
        if doANN:
        
            for i,n_hidden_units in enumerate(range(hidden_unit_start,hidden_unit_end)):

                print('Hidden units = {}'.format(n_hidden_units))
                # Extract training and test set for current CV fold, convert to tensors
                X_train = torch.tensor(X[train_index,:], dtype=torch.float)
                y_train = torch.tensor(y[train_index], dtype=torch.float)
                X_test = torch.tensor(X[test_index,:], dtype=torch.float)
                y_test = torch.tensor(y[test_index], dtype=torch.float)
                
                # Train the net on training data
                net, final_loss, learning_curve = trainANN(n_hidden_units)
                print('\n\tBest loss: {}\n'.format(final_loss))
                
                # Determine estimated class labels for test set
                y_test_est = net(X_test)
                
                # Determine errors and errors
                se = (y_test_est.float()-y_test.float())**2 # squared error
                mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
                errors_ANN_inner[k1,i] = mse
        #endregion

    # region ANN: learning on optimal hidden unit
    if doANN:
        opt_hidden_unit = np.argmin(np.mean(errors_ANN_inner,axis=1)) + hidden_unit_start
        optimal_n_hidden_units.append(opt_hidden_unit)
    
        #### Train on d_train_outer with the newly found best parameters

        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.tensor(X[train_index_o,:], dtype=torch.float)
        y_train = torch.tensor(y[train_index_o], dtype=torch.float)
        X_test = torch.tensor(X[test_index_o,:], dtype=torch.float)
        y_test = torch.tensor(y[test_index_o], dtype=torch.float)
        
        # Train the net and determine class labels
        net, final_loss, learning_curve = trainANN(opt_hidden_unit)
        y_test_est = net(X_test)
        learning_curves.append(learning_curve)
        
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        errors_ANN_outer[k2] = mse

        #Save results if these are the best so far
        if mse < mse_best_ANN:
            y_est_best_ANN = y_test_est.data.numpy() 
            y_true_best_ANN = y_test.data.numpy()
            mse_best_ANN = mse
            net_best_hidden_units = opt_hidden_unit
    # endregion


if doANN:
    print(optimal_n_hidden_units)
    print(errors_ANN_outer)
    print('\nANN: Estimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors_ANN_outer)), 4)))


    # Print the average classification error rate

##Plotting Data

#region ANN Plotting

"""
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
"""
if doANN:    
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    #Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

    for i,learning_curve in enumerate(learning_curves):
        h, = summaries_axes[0].plot(learning_curve, color=color_list[i])
        h.set_label('CV fold {0}'.format(i+1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')

        #Display the MSE across folds
        summaries_axes[1].bar(np.arange(1, K2+1), errors_ANN_outer, color=color_list)
        summaries_axes[1].set_xlabel('Outer fold')
        summaries_axes[1].set_xticks(np.arange(1, K2+1))
        summaries_axes[1].set_ylabel('MSE')
        summaries_axes[1].set_title('Test mean-squared-error')

    plt.figure(figsize=(10,10))

    y_est = y_est_best_ANN
    y_true = y_true_best_ANN

    y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
    axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Best ANN model using {} hidden units'.format(net_best_hidden_units)])
    plt.title('Attack power: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()

    plt.show()
#endregion