###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
from sklearn.cluster import KMeans
import idx2numpy
import gzip
import numpy as np
import matplotlib.pyplot as plt


def preprocessing(input):
    output = []
    for i in range(len(input)):
        output.append(input[i].flatten())
    output = np.array(output)
    return output


if __name__ == '__main__':
    file = gzip.open('fmnist_images.gz', 'r')
    img = idx2numpy.convert_from_file(file)
    file_label = gzip.open('fmnist_labels.gz', 'r')
    label = idx2numpy.convert_from_file(file_label)
    data = preprocessing(img)
    kmeans = KMeans(n_clusters=10, init="random").fit(data)
    labels = kmeans.predict(data)
    for i in range(10):
        j = 0
        fig = plt.figure()
        ax = fig.subplots(2,5)
        fig.suptitle('class' + str(i))
        while j < 10:
            random_i = np.random.randint(10000)
            if labels[random_i] == i:
                ax[int(j/5)][j%5].imshow(img[random_i], cmap=plt.get_cmap('gray'))
                j += 1
        plt.show()

