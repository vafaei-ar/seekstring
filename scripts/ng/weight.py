#! /home/gf/pakages/miniconda2/bin/python2.7


import numpy as np
import pylab as plt	
from scipy.optimize import least_squares
	
gs = np.array([[0,1],[0.3,0.1],[1.0,1.0]])
	
def chi2(var, x, y):
    return np.exp(var[0]*(x-var[1]))-y  
    
def fun(var, x):
    return np.exp(var[0]*(x-var[1]))

res_lsq = least_squares(chi2, np.array([0.0, -1.0]), args=(gs[:2,0], gs[:2,1]))
x1 = np.linspace(0,gs[1,0],1000)
y1 = fun(res_lsq['x'], x1)

res_lsq = least_squares(chi2, np.array([0.0, 1.0]), args=(gs[1:,0], gs[1:,1]))
x2 = np.linspace(gs[1,0],gs[2,0],1000)
y2 = fun(res_lsq['x'], x2)

plt.plot(gs[:,0],gs[:,1],'ro')
plt.plot(x1,y1,'b--')
plt.plot(x2,y2,'b--')

plt.show()


