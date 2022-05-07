import numpy as np
import scipy.linalg as la

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