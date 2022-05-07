###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def binary_matrix(x):
    Y = np.ones([len(x),len(x.columns)-3],dtype=int)
    mode = x.iloc[:, 3:].mode(dropna=False)
    for row in range(len(x)):
        for column in range(3, 10104):
            if x.iloc[row,column] == mode.iloc[0,column-3]:
                Y[row,column-3] = 0
    np.savetxt('Y.txt', Y, delimiter=',', fmt="%i")

if __name__ == '__main__':
#   at first run I called this function to built Y matrix and saved it in Y.txt file
    # data = pd.read_table("1KGP.txt")
    # data.columns = ['column']
    # data = data["column"].str.split(' ', expand=True)
    # binary_matrix(data)
# ###################################################################################
    Y = np.loadtxt("Y.txt", delimiter=',')
    pca = PCA(n_components=2)
    pca.fit(Y)
    vectors = pca.components_
    new_Y = np.dot(Y,vectors.T)
    print(new_Y.shape)
