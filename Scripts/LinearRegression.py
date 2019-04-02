from ProjectData import *
from matplotlib.pylab import figure, plot, subplot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm
import numpy as np

allAtt=dNorm.columns.values.tolist()
# Split dataset into features and target vector
catchRate_idx = allAtt.index('Catch_Rate')
y = dNorm.iloc[:,catchRate_idx]

X_cols = list(range(1,catchRate_idx)) + list(range(catchRate_idx+1,len(dNorm.columns.values)))
X = dNorm.iloc[:,X_cols]

"""
#Some attributes included in Linear Regression
hp_idx = allAtt.index('HP')
attack_idx = allAtt.index('Attack')
defense_idx= allAtt.index('Defense')
spdef_idx= allAtt.index('Sp_Def')
spatk_idx= allAtt.index('Sp_Atk')
speed_idx= allAtt.index('Speed')
Xhp = np.array(dNorm)[:,hp_idx].reshape(-1,1)
Xatk = np.array(dNorm)[:,attack_idx].reshape(-1,1)
Xdef=np.array(dNorm)[:,defense_idx].reshape(-1,1)
Xspdef = np.array(dNorm)[:,spdef_idx].reshape(-1,1)
Xspatk = np.array(dNorm)[:,spatk_idx].reshape(-1,1)
Xspd=np.array(dNorm)[:,speed_idx].reshape(-1,1)
X = np.asarray(np.bmat('X, Xhp, Xatk, Xdef, Xspdef, Xspatk, Xspd'))
"""

#All attributes included in Linear Regression
for name in allAtt:
    if(name!="Catch_Rate"):
        idx=allAtt.index(name)
        tmp =np.array(dNorm)[:,idx].reshape(-1,1)
        X = np.asarray(np.bmat('X, tmp')) 
    

# Fit ordinary least squares regression model
model = lm.LinearRegression()
model.fit(X,y)

# Predict Catch Rate content
y_est = model.predict(X)
residual = y_est-y

# Display plots
figure(figsize=(12,12))

subplot(4,1,1)
plot(y, y_est, '.g')
xlabel('Catch Rate (true)'); ylabel('Catch Rate (estimated)')

subplot(4,1,2)
hist(residual,40)
"""
subplot(4,3,7)
plot(Xhp, residual, '.r')
xlabel('Hp'); ylabel('Residual')

subplot(4,3,8)
plot(Xatk, residual, '.r')
xlabel('Attack'); ylabel('Residual')

subplot(4,3,9)
plot(Xdef, residual, '.r')
xlabel('Defense'); ylabel('Residual')

subplot(4,3,10)
plot(Xspdef, residual, '.r')
xlabel('Sp_Def'); ylabel('Residual')

subplot(4,3,11)
plot(Xspatk, residual, '.r')
xlabel('Sp_Atk'); ylabel('Residual')

subplot(4,3,12)
plot(Xspd, residual, '.r')
xlabel('Speed'); ylabel('Residual')
"""

show()

print('Ran Exercise 5.2.5')