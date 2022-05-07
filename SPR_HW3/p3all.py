###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi,voronoi_plot_2d
from sklearn.neighbors import KNeighborsClassifier
import random
from matplotlib.colors import ListedColormap
import seaborn as sns
import math

def distance(train_x,test):
    distances = []
    for item in train_x:
        distances.append(math.sqrt(pow(item[0] - test[0],2)+pow(item[1] - test[1],2)))
    return distances

def KNN(train_x,train_y,test,k):
    output=[]
    for t in test:
        # array_d = [3,5,6,8,2,3,55,13,0,5]
        fake = 0
        real = 0
        array_d = distance(train_x,t)
        idx = np.argpartition(array_d, k)
        for i in idx[:k]:
            if train_y[i] == 1 :
                fake += 1
            elif train_y[i] == 0:
                real += 1
        if fake > real:
            output.append(1)
        else:
            output.append(0)
    return np.array(output)


if __name__ == '__main__':
    data = []
    with open('data_banknote_authentication.txt') as my_file:
        for line in my_file:
            data.append([list(map(float, x.split(','))) for x in line.split(' ')])
    random.shuffle(data)
    # print(data)
    data = np.array(data)
######################################################################
# part a for feature selection
#     fake = []
    # real = []
    # for i in range(len(data)):
    #     if data[i][0][4] == 1:
    #         fake.append(data[i][0][:])
    #     else:
    #         real.append(data[i][0][:])
    # fake = np.array(fake)
    # real = np.array(real)
    # plt.scatter(fake[:,1],fake[:,3],color="red",label="fake")
    # plt.scatter(real[:,1],real[:,3],color="orange",label="real")
    # plt.title("feature 2 and 4 ")
    # plt.legend()
    # plt.show()
#######################################################################
# part b,c,d
    X = []
    Y = []
    for j in range(500):
        X.append([data[j][0][0],data[j][0][1]])
        Y.append(data[j][0][4])
    X = np.array(X)
    Y = np.array(Y)
    X_test = []
    Y_test = []
    h = .1
    for k in range(872):
        X_test.append([data[k+500][0][0], data[k+500][0][1]])
        Y_test.append(data[k+500][0][4])
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
# using library for having high speed
    neigh = KNeighborsClassifier(n_neighbors= 3)
    neigh.fit(X,Y)
    Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
# using my own KNN code
#     Z = KNN(X, Y,np.c_[xx.ravel(), yy.ravel()], k=3)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['orange', 'cyan'])
    cmap_bold = ['darkorange', 'c']
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    sns.scatterplot(x=X_test[:,0],y=X_test[:,1], hue= Y_test, palette=cmap_bold, alpha=1.0, edgecolor="white")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()