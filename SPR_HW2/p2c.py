###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################

import numpy as np
from numpy import log as ln

mean_1 = [1, 2]
mean_2 = [-1, -3]
cov_1 = [[1.8, -0.7], [-0.7, 1.8]]
cov_2 = [[1.5, 0.3], [0.3, 1.5]]


def mahalanobis(x,cov,mu):
    firstdot = np.dot(np.subtract(x,mu).T,cov)
    output = np.dot(firstdot, np.subtract(x,mu))
    return -0.5*output
def MDC(x,mu):
    m = 0.5 * np.dot(mu.T,mu)
    output = np.subtract(np.dot(x.T,mu),m)
    return output

def main():
    x1_w1, x2_w1 = np.random.multivariate_normal(mean_1, cov_1, 1000).T
    x1_w2, x2_w2 = np.random.multivariate_normal(mean_2, cov_2, 1000).T
    cov_1_det = np.linalg.det(cov_1)
    cov_2_det = np.linalg.det(cov_2)
    c1 = -0.5 * ln(cov_1_det)
    c2 = -0.5 * ln(cov_2_det)
    e1 = 0
    e2 = 0
    cov_1_inv = np.linalg.inv(cov_1)
    cov_2_inv = np.linalg.inv(cov_2)
    for i in range(1000):
        d1 = mahalanobis(np.array([x1_w1[i], x2_w1[i]]), cov_1_inv, mean_1) + c1
        d2 = mahalanobis(np.array([x1_w1[i], x2_w1[i]]), cov_2_inv, mean_2) + c2
        # d1 = MDC(np.array([x1_w1[i], x2_w1[i]]), np.array(mean_1))
        # d2 = MDC(np.array([x1_w1[i], x2_w1[i]]), np.array(mean_2))
        if d1 < d2:
            e1 += 1
    for j in range(1000):
        d1 = mahalanobis(np.array([x1_w2[j], x2_w2[j]]), cov_1_inv, mean_1) + c1
        d2 = mahalanobis(np.array([x1_w2[j], x2_w2[j]]), cov_2_inv, mean_2) + c2
        # d1 = MDC(np.array([x1_w2[i], x2_w2[i]]),np.array(mean_1))
        # d2 = MDC(np.array([x1_w2[i], x2_w2[i]]),np.array(mean_2))
        if d1 > d2:
            e2 += 1
    # for part c return this
    return (e1+e2) / 2000
    # for part d return this
    ###### FN = e1, FP = e2, TP = 1- e1,TN = 1 - e2
    # recall = (1000 - e1)/1000
    # precision = (1000 - e1) / (1000 - e1 + e2)
    # return (2 * precision * recall) / (precision + recall)

if __name__ == '__main__':
    print(main())