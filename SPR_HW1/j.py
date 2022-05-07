import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
import math


# Construct a 2*2 matrix P                         -----part-j------
mean = [2, 1]
cov = [[2,1], [1,3]]
x, y = np.random.multivariate_normal(mean, cov, 500).T
eigen_values, eigen_vectors = la.eig(cov)
eig_vec1 = eigen_vectors[:,0]
eig_vec2 = eigen_vectors[:,1]
P = [eig_vec2,eig_vec1]
new_x = x[:] - 2
new_y = y[:] - 1
X = np.transpose([new_x,new_y])
Y = np.dot(X,P)
Mx = 0
My = 0
for i in range(int(Y.size / 2)):
    Mx = Mx + Y[i][0]
Mx = Mx /int(Y.size / 2)
for j in range(int(Y.size / 2)):
    My = My + Y[j][1]
My = My / int(Y.size / 2)
cov11 =0
cov12 =0
cov22 =0
for t in range(int(Y.size / 2)):
    cov11 = cov11 +(Y[t][0] - Mx)*(Y[t][0] - Mx)
    cov12 = cov12 +(Y[t][0] - Mx)*(Y[t][1] - My)
    cov22 = cov22 +(Y[t][1] - My)*(Y[t][1] - My)
cov11 = cov11 / x.size
cov12 = cov12 / x.size
cov22 = cov22 / x.size
covariance = [[cov11,cov12],[cov12,cov22]]
print(covariance)
eigen_values, eigen_vectors = la.eig(covariance)
eig_vec1 = eigen_vectors[:,0]
eig_vec2 = eigen_vectors[:,1]
eigen_v, eigen_ve = la.eig(cov)
eig_1 = eigen_ve[:,0]
eig_2 = eigen_ve[:,1]
fig, axs = plt.subplots(2, constrained_layout=True)
axs[0].plot(x, y, "." , color = "red")
axs[0].arrow(mean[0].real,mean[1].real, eig_1[0], eig_1[1])
axs[0].arrow(mean[0].real,mean[1].real, eig_2[0], eig_2[1])
axs[1].plot(Y[:,0], Y[:,1], "." , color = "lime")
axs[1].arrow(0,0, eig_vec1[0], eig_vec1[1])
axs[1].arrow(0,0, eig_vec2[0], eig_vec2[1])
axs[0].set_xlabel('x1 values')
axs[0].set_ylabel('x2 values')
axs[1].set_xlabel('y1 values')
axs[1].set_ylabel('y2 values')
plt.show()