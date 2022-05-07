import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
import math


# Simultaneously diagonalise sigma and form vector V -----part-g-----
mean = [2, 1]
cov = [[2,1], [1,3]]
x, y = np.random.multivariate_normal(mean, cov, 500).T
eig_val_cov,eig_vec_cov = la.eig(cov)
P = [eig_vec_cov[0],eig_vec_cov[1]]
D = [[eig_val_cov[0],0],[0,eig_val_cov[1]]]
P_inv = np.linalg.inv(P)
print("sigma =")
print(cov)
print("D =")
print(D)
print("P =")
print(P)
print("PDP_inv = ")
print(np.dot(np.dot(P,D),P_inv))
V = np.transpose([eig_val_cov[0],eig_val_cov[1]])
# the covariance matrix of the transformed data becomes I         -----part-h&K-----
# to run this part you need to uncomment previous part

D[0][0] = math.sqrt(D[0][0].real)
D[1][1] = math.sqrt(D[1][1].real)
D_inv_root = np.linalg.inv(D)
Ww = np.dot(D_inv_root,np.transpose(P))
new_X = np.r_[[x],[y]]
Y = np.dot(Ww,new_X)
Mx = 0
My = 0
for i in range(Y[0].size):
    Mx = Mx + Y[0][i]
Mx = Mx / Y[0].size
for j in range(Y[1].size):
    My = My + Y[1][j]
My = My / Y[1].size
cov11 =0
cov12 =0
cov22 =0
for t in range(Y[0].size):
    cov11 = cov11 +(Y[0][t] - Mx)*(Y[0][t] - Mx)
    cov12 = cov12 +(Y[0][t] - Mx)*(Y[1][t] - My)
    cov22 = cov22 +(Y[1][t] - My)*(Y[1][t] - My)
cov11 = cov11 / x.size
cov12 = cov12 / x.size
cov22 = cov22 / x.size
covariance = [[cov11,cov12],[cov12,cov22]]
print(covariance)
eigen_values, eigen_vectors = la.eig(covariance)
print(eigen_values)
print(eigen_vectors)
fig, axs = plt.subplots(2, constrained_layout=True)
axs[0].plot(x, y, "." , color = "red")
axs[1].plot(Y[0], Y[1], "." , color = "lime")
axs[0].set_xlabel('x1 values')
axs[1].set_xlabel('y1 values')
axs[1].set_ylabel('y2 values')
plt.show()