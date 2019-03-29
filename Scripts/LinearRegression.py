from CarsData import *
from matplotlib.pylab import figure, plot, subplot, xlabel, ylabel, hist, show
import sklearn.linear_model as lm
import numpy as np

# Split dataset into features and target vector
catchRate_idx = attNoK.index('Catch_Rate')
y = dLinearReg.iloc[:,catchRate_idx]

X_cols = list(range(1,catchRate_idx)) + list(range(catchRate_idx+1,len(attNoK)))
X = dLinearReg.iloc[:,X_cols]

# Additional nonlinear attributes
hp_idx = attNoK.index('HP')
attack_idx = attNoK.index('Attack')
defense_idx= attNoK.index('Defense')
spdef_idx= attNoK.index('Sp_Def')
spatk_idx= attNoK.index('Sp_Atk')
speed_idx= attNoK.index('Speed')
Xhp = np.array(dLinearReg)[:,hp_idx].reshape(-1,1)
Xatk = np.array(dLinearReg)[:,attack_idx].reshape(-1,1)
Xdef=np.array(dLinearReg)[:,defense_idx].reshape(-1,1)
Xspdef = np.array(dLinearReg)[:,spdef_idx].reshape(-1,1)
Xspatk = np.array(dLinearReg)[:,spatk_idx].reshape(-1,1)
Xspd=np.array(dLinearReg)[:,speed_idx].reshape(-1,1)
X = np.asarray(np.bmat('X, Xhp, Xatk, Xdef, Xspdef, Xspatk, Xspd'))

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

show()

print('Ran Exercise 5.2.5')