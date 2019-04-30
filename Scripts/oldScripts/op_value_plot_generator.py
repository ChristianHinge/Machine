#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# example data
x = list(range(10))
y_dat = np.random.randint(0,10,(10,10))
print(y_dat)
y = np.mean(y_dat,axis=0)
yerr = np.quantile(y_dat,[0.25,0.75],axis=0)
print(yerr)

plt.figure(figsize=(6,6))

plt.boxplot(y_dat)

"""
f , axarray = plt.subplots(10, figsize=(6,6))
for i in range(10):
    axarray[i].boxplot(y_dat[:,i])
plt.errorbar(x, y, yerr=yerr, fmt='o',capsize=5,capthick=3,ms=10)
plt.grid()
plt.title('Lambda values average performance')
"""
plt.show()