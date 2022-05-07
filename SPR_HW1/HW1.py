###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
import math
###############################################
# Generate samples from three normal distributions 1-D   -----part-a-----
# mu, sigma = 5, 3 # mean and standard deviation
# s = np.random.normal(mu, sigma, 500)
# display outputs
# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
# plt.show()
###############################################
# Generate samples from three normal distributions 2-D    -----part-b,c-----
# N = 500
# X = np.linspace(-5,6, N)
# Y = np.linspace(-5,5, N)
# X, Y = np.meshgrid(X, Y)
# # Mean vector and covariance matrix
# mu = np.array([2,-1])
# Sigma = np.array([[ 3,0], [0,0.2]])
# # Pack X and Y into a single 3-dimensional array
# pos = np.empty(X.shape + (2,))
# pos[:, :, 0] = X
# pos[:, :, 1] = Y
# def multivariate_gaussian(pos, mu, Sigma):
#     n = mu.shape[0]
#     Sigma_det = np.linalg.det(Sigma)
#     Sigma_inv = np.linalg.inv(Sigma)
#     N = np.sqrt((2*np.pi)**n * Sigma_det)
#     fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
#     return np.exp(-fac / 2) / N
# # The distribution on the variables X, Y packed into pos.
# Z = multivariate_gaussian(pos, mu, Sigma)
# # Create a surface plot and projected filled contour plot under it.
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,cmap=cm.viridis)
# ax.set_xlabel('X1 values')
# ax.set_ylabel('X2 values')
# cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
# # Adjust the limits, ticks and view angle
# ax.set_zlim(-0.15,0.2)
# ax.set_zticks(np.linspace(0,0.2,5))
# ax.view_init(45,-15)
# plt.show()
###############################################
# Display the generated samples                    -----part-c3-----
# mean = [2, -1]
# cov = [[3, 0], [0, 0.2]]
# x, y = np.random.multivariate_normal(mean, cov, 500).T
# plt.plot(x, y, '.')
# plt.axis('equal')
# plt.show()

###############################################
# Calculate the sample mean M,sigma                  -----part-f-----
# mean = [2, 1]
# cov = [[2,1], [1,3]]
# x, y = np.random.multivariate_normal(mean, cov, 500).T
# Mx = 0
# My = 0
# for i in range(x.size):
#     Mx = Mx + x[i]
# Mx = Mx / x.size
# for j in range(y.size):
#     My = My + y[j]
# My = My / y.size
# cov11 =0
# cov12 =0
# cov22 =0
# for t in range(x.size):
#     cov11 = cov11 +(x[t] - Mx)*(x[t] - Mx)
#     cov12 = cov12 +(x[t] - Mx)*(y[t] - My)
#     cov22 = cov22 +(y[t] - My)*(y[t] - My)
# cov11 = cov11 / x.size
# cov12 = cov12 / x.size
# cov22 = cov22 / x.size
# covariance = [[cov11,cov12],[cov12,cov22]]
# print(covariance)
###############################################
# Simultaneously diagonalise sigma and form vector V -----part-g-----
# mean = [2, 1]
# cov = [[2,1], [1,3]]
# x, y = np.random.multivariate_normal(mean, cov, 500).T
# eig_val_cov,eig_vec_cov = la.eig(cov)
# P = [eig_vec_cov[0],eig_vec_cov[1]]
# D = [[eig_val_cov[0],0],[0,eig_val_cov[1]]]
# P_inv = np.linalg.inv(P)
# print("sigma =")
# print(cov)
# print("D =")
# print(D)
# print("P =")
# print(P)
# print("PDP_inv = ")
# print(np.dot(np.dot(P,D),P_inv))
# V = np.transpose([eig_val_cov[0],eig_val_cov[1]])
###############################################
# the covariance matrix of the transformed data becomes I         -----part-h&K-----
# to run this part you need to uncomment previous part

# D[0][0] = math.sqrt(D[0][0].real)
# D[1][1] = math.sqrt(D[1][1].real)
# D_inv_root = np.linalg.inv(D)
# Ww = np.dot(D_inv_root,np.transpose(P))
# new_X = np.r_[[x],[y]]
# Y = np.dot(Ww,new_X)
# Mx = 0
# My = 0
# for i in range(Y[0].size):
#     Mx = Mx + Y[0][i]
# Mx = Mx / Y[0].size
# for j in range(Y[1].size):
#     My = My + Y[1][j]
# My = My / Y[1].size
# cov11 =0
# cov12 =0
# cov22 =0
# for t in range(Y[0].size):
#     cov11 = cov11 +(Y[0][t] - Mx)*(Y[0][t] - Mx)
#     cov12 = cov12 +(Y[0][t] - Mx)*(Y[1][t] - My)
#     cov22 = cov22 +(Y[1][t] - My)*(Y[1][t] - My)
# cov11 = cov11 / x.size
# cov12 = cov12 / x.size
# cov22 = cov22 / x.size
# covariance = [[cov11,cov12],[cov12,cov22]]
# print(covariance)
# eigen_values, eigen_vectors = la.eig(covariance)
# print(eigen_values)
# print(eigen_vectors)
# fig, axs = plt.subplots(2, constrained_layout=True)
# axs[0].plot(x, y, "." , color = "red")
# axs[1].plot(Y[0], Y[1], "." , color = "lime")
# axs[0].set_xlabel('x1 values')
# axs[1].set_xlabel('y1 values')
# axs[1].set_ylabel('y2 values')
# plt.show()

###############################################
# Calculate the eigenvalues and eigenvectors         -----part-i-----
# mean = [2, 1]
# cov = [[2,1], [1,3]]
# x, y = np.random.multivariate_normal(mean, cov, 500).T
# eigen_values, eigen_vectors = la.eig(cov)
# eig_vec1 = eigen_vectors[:,0]
# eig_vec2 = eigen_vectors[:,1]
# plt.plot(x, y, "." , color = "orange")
# plt.arrow(mean[0].real,mean[1].real, eig_vec1[0], eig_vec1[1])
# plt.arrow(mean[0].real,mean[1].real, eig_vec2[0], eig_vec2[1])
# plt.show()
###############################################
# # Construct a 2*2 matrix P                         -----part-j------
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