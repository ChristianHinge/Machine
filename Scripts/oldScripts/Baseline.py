from ProjectData import *
from matplotlib.pylab import *
import sklearn.linear_model as lm
from sklearn import model_selection
import numpy as np
from toolbox_02450 import rlr_validate

allAtt=dNorm.columns.values.tolist()
# Split dataset into features and target vector
# Attack is predicted. Attack is continous ratio in the interval 5 to 165. 
attack_idx = allAtt.index('Attack')
y = dNorm.iloc[:,attack_idx]

X_cols = list(range(0,attack_idx)) + list(range(attack_idx+1,len(dNorm.columns.values)))
X = dNorm.iloc[:,X_cols]


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]
    k+=1

# Display plots
figure(figsize=(12,12))

subplot(4,1,1)
plot(baseline)
xlabel('Attack (true)'); ylabel('Attack (estimated)')



show()

