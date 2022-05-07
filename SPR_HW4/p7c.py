###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import idx2numpy
import gzip
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


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
    data = preprocessing(img)
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    pca.fit(data)
    eigenvectors = pca.components_
    Y = np.dot(data,eigenvectors.T)
    kmeans = KMeans(n_clusters=7, init="random").fit(Y)
    labels = kmeans.predict(Y)
    for i,data in enumerate(Y):
        plt.scatter(data[0],data[1],color=color[labels[i]],s=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', s=30)
    plt.title('Kmeans')
    plt.show()