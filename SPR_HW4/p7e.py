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

def init_centroids(Y,label,n):
    c = np.zeros([10, n])
    counter = np.zeros([10, 1])
    for i in range(len(Y)):
        if label[i] == 0:
            c[0] += Y[i]
            counter[0] += 1
        elif label[i] == 1:
            c[1] += Y[i]
            counter[1] += 1
        elif label[i] == 2:
            c[2] += Y[i]
            counter[2] += 1
        elif label[i] == 3:
            c[3] += Y[i]
            counter[3] += 1
        elif label[i] == 4:
            c[4] += Y[i]
            counter[4] += 1
        elif label[i] == 5:
            c[5] += Y[i]
            counter[5] += 1
        elif label[i] == 6:
            c[6] += Y[i]
            counter[6] += 1
        elif label[i] == 7:
            c[7] += Y[i]
            counter[7] += 1
        elif label[i] == 8:
            c[8] += Y[i]
            counter[8] += 1
        elif label[i] == 9:
            c[9] += Y[i]
            counter[9] += 1
    for j in range(10):
        c[j, :] = c[j, :] / counter[j]
    return c

def accuracy(true,cluster):
    counter = 0
    for i in range(len(true)):
        if true[i] == cluster[i]:
            counter += 1
    return counter/10000

if __name__ == '__main__':
    file = gzip.open('fmnist_images.gz', 'r')
    img = idx2numpy.convert_from_file(file)
    file_label = gzip.open('fmnist_labels.gz', 'r')
    label = idx2numpy.convert_from_file(file_label)
    data = preprocessing(img)
    # for i in range(2, 748):
    #     pca = PCA(n_components=i)
    #     Y = pca.fit_transform(data)
    #     if sum(pca.explained_variance_ratio_ ) > 0.95:
    #         print(sum(pca.explained_variance_ratio_ ))
    #         print(i)
    #         break
    #
    #
    fig = plt.figure()
    ax = fig.subplots(2, 3)
    pca = PCA(n_components=185)
    Y = pca.fit_transform(data)
    for i in range(3):
        random = np.random.randint(10000)
        ax[0][i].imshow(img[random], cmap=plt.get_cmap('gray'))
        ax[0][i].set_title('before PCA')
        ax[1][i].imshow(np.array(pca.inverse_transform(Y[random])).reshape(-1,28),cmap=plt.get_cmap('gray'))
        ax[1][i].set_title('after PCA')

    plt.show()