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
    kmeans = KMeans(n_clusters=10, init="random").fit(data)
    labels = kmeans.predict(data)
    N = 10
    arr = np.zeros([10,10])
    for i in range(10):
        for j in range(10):
            arr[i,j] = len(([data[k] for k in range(len(data)) if labels[k] == i and label[k] == j]))
    d = tuple(map(tuple, arr))
    ind = np.arange(N)
    width = 0.1
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(ind, d[0], width, color='red')
    ax.bar(ind, d[1], width, color='blue')
    ax.bar(ind, d[2], width, color='yellow')
    ax.bar(ind, d[3], width, color='pink')
    ax.bar(ind, d[4], width, color='orange')
    ax.bar(ind, d[5], width, color='green')
    ax.bar(ind, d[6], width, color='m')
    ax.bar(ind, d[7], width, color='gray')
    ax.bar(ind, d[8], width, color='lightgreen')
    ax.bar(ind, d[9], width, color='violet')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(ind, ('0', '1', '2', '3', '4', '5','6','7','8','9'))
    ax.set_yticks(np.arange(0, 81, 10))
    ax.legend(labels=['0', '1', '2', '3', '4', '5','6','7','8','9'])
    plt.show()

# bar plot for each cluster
    for j in range(10):
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        width = 0.2
        N = 10
        arr = np.zeros([1, 10])
        for i in range(10):
            length = len(([data[k] for k in range(len(labels)) if labels[k] == j and label[k] == i]))
            c_le = len(([data[t] for t in range(len(labels)) if labels[t] == j]))
            arr[0, i] = length
            # print(length)
            ax.text(i, length, str(length / c_le )[0:4] + '%')
        d = tuple(map(tuple, arr))
        ind = ['C1', 'C2', 'C3', 'C4', 'C5' , 'C6' , 'C7','C8','C9','C10']
        ax.bar(ind, d[0], width, color=color[j])
        fig.suptitle('cluster' + str(j+1))
        plt.show()