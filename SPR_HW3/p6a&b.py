###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing



def MSE(a,x):
    # print(x)
    b = 0.02 * np.ones(len(x))
    q = np.dot(x,a) - b
    output = np.dot(q.T,q)
    return output


def gradian(x,y):
    a = np.array([0, 0, 0])
    b = 0.02 * np.ones(len(y))
    i = 0
    x.insert(0, "extra", np.ones(len(x)), True)
    for item in y:
        if item == 'Gentoo':
            x.iloc[i] = -1 * x.iloc[i]
        i += 1
    x = x.to_numpy()
    hessian = 2 * np.dot(x.T,x)
    H = np.linalg.inv(hessian)
    J_a = MSE(a,x)
    iteration = 0
    while True:
# Gradient  with different learning rate(part b and a)
        a_k = a - 0.1 * np.dot(x.T,(np.dot(x,a) - b)) / len(x)
        # a_k = a - 10 * np.dot(x.T,(np.dot(x,a) - b)) / len(x)
        # a_k = a - 0.01 * np.dot(x.T,(np.dot(x,a) - b)) / len(x)
#         a_k = a - 1 * np.dot(x.T,(np.dot(x,a) - b)) / len(x)
#         a_k = a - 5 * np.dot(x.T,(np.dot(x,a) - b)) / len(x)
# Neowon
#         a_k = a - np.dot(H,np.dot(x.T,(np.dot(x,a) - b)))
        new_J = MSE(a_k,x)
        if new_J <= J_a :
            J_a = new_J
            a = a_k
        else:
            break
        iteration += 1
    print(iteration)
    return a


def test_phase(a,x):
    label = []
    x = x.to_numpy()
    for item in x:
        g = item[0] * a[1] + item[1] * a[2] + a[0]
        if g >= 0:
            label.append('Adelie')
        else:
            label.append('Gentoo')
    return np.array(label)

if __name__ == '__main__':
    dataset = pd.read_csv('penguins.csv')
# preprocessing
    for i in range(len(dataset)):
        if np.isnan(dataset.iloc[i]['bill_length_mm']):
            dataset.iloc[i, dataset.columns.get_loc('bill_length_mm')]= min(dataset['bill_length_mm'])
        if np.isnan(dataset.iloc[i]['bill_depth_mm']):
            dataset.iloc[i, dataset.columns.get_loc('bill_depth_mm')]= min(dataset['bill_depth_mm'])
# normalization
    d = dataset[['bill_length_mm', 'bill_depth_mm']].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(d)
    d = pd.DataFrame(x_scaled)
    d.insert(2,'species',dataset['species'])
    dataset = d.rename(columns={0 :'bill_length_mm',1:'bill_depth_mm'})
# shuffle data
    dataset = shuffle(dataset)
    train = dataset.iloc[:300]
    test = dataset.iloc[300:]
    train = train.loc[train['species'] != 'Chinstrap']
    test = test.loc[test['species'] != 'Chinstrap']
    X_train = train[['bill_length_mm', 'bill_depth_mm']]
    Y_train = train['species']
    X_test = test[['bill_length_mm', 'bill_depth_mm']]
    Y_test = test['species']
    a = gradian(X_train.copy(),Y_train)
    print(a)
    p = np.array([0, 1])
    y = -1 * a[1] / a[2] * p + a[0]
    predicted = test_phase(a, X_test)
    i = 0
    error = 0
    Y_test = Y_test.to_numpy()
    for l in predicted:
        if l != Y_test[i]:
            error += 1
        i += 1
    print(error / len(predicted))
    fig, ax = plt.subplots(2)
    ax[1].scatter(test.loc[test['species'] == 'Adelie'].to_numpy()[:, 0], test.loc[test['species'] == 'Adelie'].to_numpy()[:, 1], color='violet', label='Adelie ')
    ax[0].scatter(train.loc[train['species'] == 'Adelie'].to_numpy()[:, 0], train.loc[train['species'] == 'Adelie'].to_numpy()[:, 1], color='yellow', label='Adelie')
    ax[0].scatter(train.loc[train['species'] == 'Gentoo'].to_numpy()[:, 0], train.loc[train['species'] == 'Gentoo'].to_numpy()[:, 1], color='orange', label='Gentoo ')
    ax[1].scatter(test.loc[test['species'] == 'Gentoo'].to_numpy()[:, 0], test.loc[test['species'] == 'Gentoo'].to_numpy()[:, 1], color='lightblue', label='Gentoo ')
    # plt.scatter(X_train.to_numpy()[:, 0], X_train.to_numpy()[:, 1], color='lightblue', label='train data')
    ax[0].plot(p, y, color='pink')
    ax[1].plot(p, y, color='pink')
    ax[0].set_title("train")
    ax[1].set_title("test")
    ax[0].legend()
    ax[1].legend()
    # fig.suptitle('Newton')
    fig.suptitle('Gradient Descent')
    plt.show()
#