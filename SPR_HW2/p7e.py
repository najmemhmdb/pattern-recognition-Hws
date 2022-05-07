###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################

import numpy as np
import matplotlib.pyplot as plt
import math

def posterior(input_array, k,m,omega):
    first = math.comb(m-1,k-1)
    second = math.comb(omega-1,k-1)
    c = (1/first) - (1/second)
    output = []
    for i in range(len(input_array)):
        output.append((k-1)/(k*math.comb(input_array[i],k)*c))

    return np.array(output)

if __name__ == '__main__':
    k = 25
    m = 200
    omega = 10000
    n = np.arange(m, omega+1,1)
    posterior = posterior(n , k,m,omega)
    plt.plot(n,posterior)
    plt.xlabel("n")
    plt.ylabel("posterior")
    plt.show()