import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as la
import math


# Calculate the eigenvalues and eigenvectors         -----part-i-----
mean = [2, 1]
cov = [[2,1], [1,3]]
x, y = np.random.multivariate_normal(mean, cov, 500).T
eigen_values, eigen_vectors = la.eig(cov)
eig_vec1 = eigen_vectors[:,0]
eig_vec2 = eigen_vectors[:,1]
plt.plot(x, y, "." , color = "orange")
plt.arrow(mean[0].real,mean[1].real, eig_vec1[0], eig_vec1[1])
plt.arrow(mean[0].real,mean[1].real, eig_vec2[0], eig_vec2[1])
plt.show()