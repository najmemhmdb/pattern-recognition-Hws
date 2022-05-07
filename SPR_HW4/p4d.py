###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def binary_matrix(x):
    Y = np.ones([len(x),len(x.columns)-3],dtype=int)
    mode = x.iloc[:, 3:].mode(dropna=False)
    for row in range(len(x)):
        for column in range(3, 10104):
            if x.iloc[row,column] == mode.iloc[0,column-3]:
                Y[row,column-3] = 0
    np.savetxt('Y.txt', Y, delimiter=',', fmt="%i")


# color = {'ACB':'red','ASW':'blue','ESN':'orange','MSL':'lightgreen','GWD':'pink','LWK':'y','YRI':'m'}
color = {'1':'blue','2':'red'}
if __name__ == '__main__':
#   at first run I called this function to built Y matrix and saved it in Y.txt file
    data = pd.read_table("1KGP.txt")
    data.columns = ['column']
    data = data["column"].str.split(' ', expand=True)
    # binary_matrix(data)
    population = np.array(data[2])
    sex = np.array(data[1])
# ###################################################################################
    Y = np.loadtxt("Y.txt", delimiter=',')
    pca = PCA(n_components=3)
    pca.fit(Y)
    vectors = pca.components_
    vectors = np.array([vectors[0,:],vectors[2,:]])
    new_Y = np.dot(Y,vectors.T)
    for pop in color.keys():
        # data = np.array([new_Y[i] for i in range(len(new_Y)) if population[i] == pop])
        data = np.array([new_Y[i] for i in range(len(new_Y)) if sex[i] == pop])
        plt.scatter(data[:,0],data[:,1],color=color.get(pop),s=5,label=pop)
    plt.title('Sex PCA 1,3')
    plt.legend()
    plt.show()
