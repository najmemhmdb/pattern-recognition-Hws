###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import idx2numpy
import gzip
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def preprocessing(input):
    output = []
    for i in range(len(input)):
        output.append(input[i].flatten())
    output = np.array(output)
    return output


color = ['red','green','blue','yellow','y','m','pink','orange','lightblue','lightgreen']

if __name__ == '__main__':
    file = gzip.open('fmnist_images.gz', 'r')
    img = idx2numpy.convert_from_file(file)
    file_label = gzip.open('fmnist_labels.gz', 'r')
    label = idx2numpy.convert_from_file(file_label)
    data = preprocessing(img)
    data = StandardScaler().fit_transform(data)
################################################################
    # pca = PCA(n_components=20)
    # pca.fit(data)
    # eigenvalues = pca.explained_variance_
    # print(eigenvalues)
    # plt.scatter(range(20),eigenvalues,s=4)
    # plt.show()
################################################################
    # pca = PCA(n_components=2)
    # pca.fit(data)
    # eigenvectors = pca.components_
    # Y = np.dot(data,eigenvectors.T)
    # for i,data in enumerate(Y):
    #     plt.scatter(data[0],data[1],color=color[label[i]],s=0.5 )
    # # plt.scatter(Y[:,0],Y[:,1],color='red',s=1)
    # plt.title("plot fashion_MNIST 2 PCA")
    # plt.xlabel('first PC')
    # plt.ylabel('second PC')
    # plt.show()
################################################################
    pca = PCA(n_components=3)
    Y = pca.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        u = np.array([Y[k] for k in range(len(Y)) if label[k] == i])
        ax.scatter(u[:,0],u[:,1],u[:,2],color=color[i],s=0.5)
    # ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], marker=".",s=0.5, color="red")
    ax.set_xlabel('First PC')
    ax.set_ylabel('Second PC')
    ax.set_zlabel('third PC')
    plt.title("plot fashion_MNIST 3 PCA")
    plt.show()