###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

def calculate_error(true,labels):
    error = 0
    for i in range(len(true)):
        if true[i] == labels[i]:
            error += 1
    return error/len(true)

def knn_model_selection(x_t,y_t,x,y):
    e_t = []
    e = []
    for i in range(20):
        neigh = KNeighborsClassifier(n_neighbors=i+1)
        neigh.fit(x_t, y_t)
        labels_validation = neigh.predict(x)
        labels_train = neigh.predict(x_t)
        e.append([i+1, calculate_error(y,labels_validation)])
        e_t.append([i+1, calculate_error(y_t,labels_train)])
    e = np.array(e)
    e_t= np.array(e_t)
    plt.plot(e[:,0],e[:,1],color="red",label="validation accuracy")
    plt.plot(e_t[:,0],e_t[:,1],color="orange",label="train accuracy")
    plt.legend()
    plt.show()

def mag(vec):
    sum = 0
    for i in vec:
        sum += pow(i,2)
    return math.sqrt(sum)


def cosine(train_x,t):
    distances=[]
    for item in train_x:
        distances.append(np.dot(item,t)/(mag(item)*mag(t)))
    return distances



def KNN(train_x,train_y,test,k):
    output=[]
    k = -1 *k
    for t in test:
        fake = 0
        real = 0
        array_d = cosine(train_x,t)
        # print(array_d)
        # print(max(array_d))
        idx = np.argpartition(array_d, k)
        for i in idx[k:]:
            if int(train_y[i]) == 1:
                fake += 1
            elif int(train_y[i]) == 0:
                real += 1
        # print(fake , real )print("----------------------------")
        if fake > real:
            # print("in fake ")
            output.append(1)
        else:
            output.append(0)
        # print("----------------------------")
    return np.array(output)


if __name__ == '__main__':
    fake = open('clean_fake.txt', 'r')
    real = open('clean_real.txt', 'r')
    # fake = open('fake_test.txt', 'r')
    # real = open('real_test.txt', 'r')
    Lines = fake.readlines()
    headers = []
    for line in Lines:
         headers.append([line.strip(),1])
    Lines = real.readlines()
    for line in Lines:
        headers.append([line.strip(),0])
    random.shuffle(headers)
    headers = np.array(headers)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(headers[:,0])
    X_train, X_rest, y_train, y_rest = train_test_split( X.toarray(), headers[:,1], test_size = 0.3, random_state = 42)
    X_validation,X_test,Y_validation,Y_test = train_test_split(X_rest,y_rest,test_size=0.5,random_state=42)
    knn_model_selection(X_train,y_train,X_validation,Y_validation)
    # Z = KNN(X_train,y_train,X_rest,k=3)
    # error = 0
    # for i in range(len(Z)):
    #     if Z[i] != int(y_rest[i]):
    #         error += 1
    # print(error/len(Z))
    # np.savetxt("output.txt",Z)