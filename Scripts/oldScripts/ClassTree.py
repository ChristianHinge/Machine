import numpy as np
from ProjectData import *
from sklearn import model_selection, tree
from platform import system
from os import getcwd
import matplotlib.pyplot as plt
from toolbox_02450 import windows_graphviz_call
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show, boxplot
from matplotlib.image import imread
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show
import graphviz as gv

target='isLegendary'

# Class indices
y = np.array(dOriginal[target])

dOriginal=dOriginal.drop(target,axis=1)
dLogReg=dLogReg.drop(target,axis=1)
#dLogReg=dLogReg.drop("Catch_Rate",axis=1)
# Names of data objects
dataobjectNames = list(dOriginal['Name'])


# Attribute names
attributeNames = list(dLogReg)


# Attribute values
X = np.asarray(np.array(dLogReg))





# Class names
classNames = ['Is not legendary', 'Is legendary']
    
# Number data objects, attributes, and classes
N, M = X.shape
C = len(classNames)

criterion='gini'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=2)
dtc = dtc.fit(X,y)

fname='tree_Legendary'
# Export tree graph .gvz file to parse to graphviz
out = tree.export_graphviz(dtc, out_file=fname + '.dot', feature_names=attributeNames)

#if system() == 'Windows':
#    windows_graphviz_call(fname=fname,
#                              cur_dir=getcwd(),
#                              path_to_graphviz=r'C:\Users\Erik Gylling\Desktop\DTU\4.Semester\IntroductionToMachineLearningAndDataMining\PythonPakker\graphviz-2.38\release')

#Kør kommandoen "dot -Tpng tree_Legendary.dot -o tree_Legendary.png"
# fra working directory for at se plot


# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# K-fold crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variable
Error_train = np.empty((len(tc),K))
Error_test = np.empty((len(tc),K))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    for i, t in enumerate(tc):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=t)
        dtc = dtc.fit(X_train,y_train.ravel())
        y_est_test = dtc.predict(X_test)
        y_est_train = dtc.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        misclass_rate_test = np.sum(y_est_test != y_test) / float(len(y_est_test))
        misclass_rate_train = np.sum(y_est_train != y_train) / float(len(y_est_train))
        Error_test[i,k], Error_train[i,k] = misclass_rate_test, misclass_rate_train
    k+=1

bestDepth=np.argmin(Error_test.mean(1))+2
dtc = tree.DecisionTreeClassifier(criterion=criterion, max_depth=bestDepth)
dtc = dtc.fit(X,y)
out = tree.export_graphviz(dtc, out_file=fname + '.dot', feature_names=attributeNames)

f = figure()
boxplot(Error_test.T)
xlabel('Model complexity (max tree depth)')
ylabel('Test error across CV folds, K={0})'.format(K))

f = figure()
plot(tc, Error_train.mean(1))
plot(tc, Error_test.mean(1))
xlabel('Model complexity (max tree depth)')
ylabel('Error (misclassification rate, CV K={0})'.format(K))
legend(['Error_train','Error_test'])

plt.title('Min. Test Error of {0}% is found on depth {1}'.format(np.round(np.min(Error_test.mean(1))*100,2),bestDepth))
show()
#plt.savefig("../Figures/bestDepth.png")
print("Error test with depth {0}:\n{1}".format(bestDepth,Error_test[bestDepth-2,:].T))

