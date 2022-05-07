import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
import math

# Calculate the sample mean M,sigma                  -----part-f-----
mean = [2, 1]
cov = [[2,1], [1,3]]
x, y = np.random.multivariate_normal(mean, cov, 500).T
Mx = 0
My = 0
for i in range(x.size):
    Mx = Mx + x[i]
Mx = Mx / x.size
for j in range(y.size):
    My = My + y[j]
My = My / y.size
cov11 =0
cov12 =0
cov22 =0
for t in range(x.size):
    cov11 = cov11 +(x[t] - Mx)*(x[t] - Mx)
    cov12 = cov12 +(x[t] - Mx)*(y[t] - My)
    cov22 = cov22 +(y[t] - My)*(y[t] - My)
cov11 = cov11 / x.size
cov12 = cov12 / x.size
cov22 = cov22 / x.size
covariance = [[cov11,cov12],[cov12,cov22]]
print(covariance)