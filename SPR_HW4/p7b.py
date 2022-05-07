###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    file_label = gzip.open('fmnist_labels.gz', 'r')
    label = idx2numpy.convert_from_file(file_label)
    data = preprocessing(img)
    data = StandardScaler().fit_transform(data)
    lda = LinearDiscriminantAnalysis(n_components=1)
    Y = lda.fit_transform(data,label)
    for i,data in enumerate(Y):
        plt.scatter(data  , 5 ,color=color[label[i]])
    plt.show()
