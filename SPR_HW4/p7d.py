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

def init_centroids(Y,label,k):
    c = np.zeros([k, 2])
    counter = np.zeros([k, 1])
    if k == 4:
        for i in range(len(Y)):
            if label[i]+1 in {1,3,5,7}:
                # print("1 3 5 7  ***** ", label[i]+1)
                c[0] += Y[i]
                counter[0] += 1
            elif label[i]+1 in {2,4}:
                # print("2,4 ****",label[i]+1)
                c[1] += Y[i]
                counter[1] += 1
            elif label[i]+1 in {6,8,10}:
                c[2] += Y[i]
                counter[2] += 1
            elif label[i]+1 in {9}:
                c[3] += Y[i]
                counter[3] += 1
        for j in range(k):
            c[j,:] = c[j,:] / counter[j]
    elif k == 7:
        for i in range(len(Y)):
            if label[i]+1 in {1,3,5}:
                c[0] += Y[i]
                counter[0] += 1
            elif label[i]+1 in {2}:
                c[1] += Y[i]
                counter[1] += 1
            elif label[i]+1 in {4}:
                c[2] += Y[i]
                counter[2] += 1
            elif label[i]+1 in {8,10}:
                c[3] += Y[i]
                counter[3] += 1
            elif label[i]+1 in {6}:
                c[4] += Y[i]
                counter[4] += 1
            elif label[i]+1 in {7}:
                c[5] += Y[i]
                counter[5] += 1
            elif label[i]+1 in {9}:
                c[6] += Y[i]
                counter[6] += 1
        for j in range(k):
            c[j,:] = c[j,:] / counter[j]
    elif k == 10:
        for i in range(len(Y)):
            if label[i]+1 in {1}:
                c[0] += Y[i]
                counter[0] += 1
            elif label[i]+1 in {2}:
                c[1] += Y[i]
                counter[1] += 1
            elif label[i]+1 in {3}:
                c[2] += Y[i]
                counter[2] += 1
            elif label[i]+1 in {4}:
                c[3] += Y[i]
                counter[3] += 1
            elif label[i]+1 in {5}:
                c[4] += Y[i]
                counter[4] += 1
            elif label[i]+1 in {6}:
                c[5] += Y[i]
                counter[5] += 1
            elif label[i]+1 in {7}:
                c[6] += Y[i]
                counter[6] += 1
            elif label[i]+1 in {8}:
                c[7] += Y[i]
                counter[7] += 1
            elif label[i]+1 in {9}:
                c[8] += Y[i]
                counter[8] += 1
            elif label[i]+1 in {10}:
                c[9] += Y[i]
                counter[9] += 1
        for j in range(k):
            c[j,:] = c[j,:] / counter[j]
    return c


color = ['red','green','blue','yellow','y','m','pink','orange','lightblue','lightgreen']

if __name__ == '__main__':
    file = gzip.open('fmnist_images.gz', 'r')
    img = idx2numpy.convert_from_file(file)
    file_label = gzip.open('fmnist_labels.gz', 'r')
    label = idx2numpy.convert_from_file(file_label)
    data = preprocessing(img)
    data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    pca.fit(data)
    eigenvectors = pca.components_
    Y = np.dot(data,eigenvectors.T)
    k = 10
    centroids_init = init_centroids(Y,label,k)
    print(centroids_init)
    kmeans = KMeans(n_clusters=k, init=centroids_init).fit(Y)
    labels = kmeans.predict(Y)
    for i,data in enumerate(Y):
        plt.scatter(data[0],data[1],color=color[labels[i]],s=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', s=30)
    plt.title('Kmeans')
    plt.show()