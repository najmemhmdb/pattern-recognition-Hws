###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def check(a,x):
    z = np.dot(a.T,x.T)
    index = []
    for i in range(len(z)):
        if z[i] <= 0:
            index.append(i)
    return np.array(index)


def perceptron(x,y):
    a = np.array([0 , 0 ,0])
    i = 0
    x.insert(0, "extra", np.ones(len(x)), True)
    for item in y:
        if item == 'Gentoo':
            x.iloc[i] = -1 * x.iloc[i]
        i += 1
    x = x.to_numpy()
    misclassified = check(a,x)
    count = 0
    while len(misclassified) != 0:
        count += 1
        Y = np.zeros(3)
        for j in misclassified:
            Y += x[j]
        a = a + 0.00001 * Y
        misclassified = check(a,x)
    print(count)
    return a

def test_phase(a,x):
    label = []
    x = x.to_numpy()
    for item in x:
        g = item[0] * a[1] + item[1] * a[2] + a[0]
        if g >= 0:
            label.append('Chinstrap')
        else:
            label.append('Gentoo')
    return np.array(label)


if __name__ == '__main__':
    dataset = pd.read_csv('penguins.csv')
    dataset = dataset.loc[dataset['species'] != 'Adelie' ]
    dataset = shuffle(dataset)
    dataset = dataset[['bill_length_mm', 'bill_depth_mm','species']].to_numpy()
    class1 = []
    class2 = []
    for item in dataset:
        if item[2] == 'Gentoo':
            class1.append(list(item))
        else:
            class2.append(list(item))
    class1 = np.array(class1)
    class2 = np.array(class2)
    # print(np.array(class1)[:,0])
    # X_train, X_test, Y_train, Y_test = train_test_split(dataset[['bill_length_mm', 'bill_depth_mm']], dataset['species'], test_size = 0.3, random_state = 42)
    # a = perceptron(X_train.copy(),Y_train)
    # p = np.array([30 , 60])
    # y = -1 * a[1]/a[2] * p + a[0]
    # predicted = test_phase(a,X_test)
    # i = 0
    # error = 0
    # Y_test= Y_test.to_numpy()
    # for l in predicted:
    #     if l != Y_test[i]:
    #         error += 1
    #     i += 1
    # print(error/len(predicted))
    # plt.scatter(X_test.to_numpy()[:,0],X_test.to_numpy()[:,1],color='violet',label='test data')
    plt.scatter(class1[:,0],class1[:,1],color='violet',label='Gentoo')
    # plt.scatter(X_train.to_numpy()[:,0],X_train.to_numpy()[:,1],color='lightblue',label='train data')
    plt.scatter(class2[:,0],class2[:,1],color='lightblue',label='Chinstrap')
    # plt.plot(p,y,color='pink',label='separator line')
    plt.legend()
    plt.show()