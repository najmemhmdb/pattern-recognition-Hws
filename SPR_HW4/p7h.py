###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import idx2numpy
import gzip
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def preprocessing(input):
    output = []
    for i in range(len(input)):
        output.append(input[i].flatten())
    output = np.array(output)
    return output

color = ['red','blue','orange','lightgreen','pink','y','m','lightblue','green','violet']
if __name__ == '__main__':
    file = gzip.open('fmnist_images.gz', 'r')
    img = idx2numpy.convert_from_file(file)
    file_label = gzip.open('fmnist_labels.gz', 'r')
    label = idx2numpy.convert_from_file(file_label)
    data = preprocessing(img)
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=185)
    Y = pca.fit_transform(data)
    kmeans = KMeans(n_clusters=10, init="random").fit(Y)
    labels = kmeans.predict(Y)


# scatter 2 features
    #
    #
    # for i in range(10):
    #     d = np.array([Y[j] for j in range(len(labels)) if labels[j] == i])
    #     plt.scatter(d[:,0],d[:,1], color=color[i])
    # plt.legend(['0', '1', '2', '3', '4', '5','6','7','8','9'])
    # plt.title('feature 1 and 2')
    # plt.show()


# scatter 3 features

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        u = np.array([Y[k] for k in range(len(Y)) if labels[k] == i])
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], color=color[i],s=1)
    ax.set_xlabel('First PC')
    ax.set_ylabel('Second PC')
    ax.set_zlabel('third PC')
    plt.title("plot fashion_MNIST 185 PCA and k-means clustering")
    plt.legend(['0', '1', '2', '3', '4', '5','6','7','8','9'])
    plt.show()
