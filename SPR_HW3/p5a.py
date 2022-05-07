###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mu1 = np.array([-3, 0])
    mu2 = np.array([3, 0])
    cov1 = np.array([1.5,1,1,1.5]).reshape(2,2)
    cov2 = np.array([1.5,-1,-1,1.5]).reshape(2,2)
    x1_w1, x2_w1 = np.random.multivariate_normal(mu1, cov1, 500).T
    x1_w2, x2_w2 = np.random.multivariate_normal(mu2, cov2, 500).T
    x1 = np.linspace(-5, 5, 100)
    x = np.linspace(-5, 5, 100)
    y = -2.78 * x + 7.41
    plt.plot(x, y, '-r', label='separating line')
    plt.scatter(x1_w1,x2_w1,color='orange',label='class 1')
    plt.scatter(x1_w2,x2_w2,color='pink',label='class 2')
    plt.xlabel('x1', color='#1C2833')
    plt.ylabel('x2', color='#1C2833')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
