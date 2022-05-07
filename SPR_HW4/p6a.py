###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    data = pd.read_csv('ALS_train.csv')
    attr = list(data.columns)
    print(attr)
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    mean_6 = np.mean(data[:,6])
    x_mu = data[:,6] - mean_6
    dic = {}
    for ind in range(data.shape[1]):
        mean = np.mean(data[:, ind])
        q = data[:,ind] - mean
        cov = np.dot(q,x_mu)
        dic.update({str(ind):abs(cov)})
    sorted_values = sorted(dic.values(),reverse=True)  # Sort the values
    sorted_dict = {}
    for i in sorted_values:
        for k in dic.keys():
            if dic[k] == i:
                sorted_dict[k] = dic[k]
                break
    keys = list(sorted_dict.keys())
    for i in range(20):
        plt.scatter(int(keys[i]),sorted_dict.get(keys[i]),color='red')
        # print(attr[int(keys[i])],sorted_dict.get(keys[i]))
        plt.annotate(attr[int(keys[i])],(int(keys[i]),sorted_dict.get(keys[i])))
    for j in range(11):
        print(attr[int(keys[j])])
    plt.show()
