###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import pandas as pd
import numpy as np

def jaccard_distance(a,b):
    A = set(a)
    B = set(b)
    j = len(A.intersection(B))/len(A.union(B))
    return 1 - j

def cluster_distance(id, ids,data):
    dis = 0
    for i in ids:
        if i != id:
            dis += jaccard_distance(str.split(data.get(str(id))),str.split(data.get(str(i))))
    return dis

def assignment(data,centroids):
    data_cluster = {}
    for centroid in centroids:
        d = []
        data_cluster.update({centroid:d})
    for id in data.keys():
        cen = 0
        min = 1
        for centroid in centroids:
            j_d = jaccard_distance(str.split(data.get(str(id))),str.split(data.get(str(centroid))))
            if j_d<min:
                min = j_d
                cen = centroid
        cluster = data_cluster.get(cen)
        cluster.append(id)
        data_cluster[cen] = cluster
    new_centroids = []
    for key in data_cluster.keys():
        ids = data_cluster.get(key)
        min = 251
        new_centroid = 0
        for tweet_id in ids:
            total_dis = cluster_distance(tweet_id,ids,data)
            if total_dis < min:
                min = total_dis
                new_centroid = tweet_id
        new_centroids.append(new_centroid)
    change = centroids == new_centroids
    if change.__class__ == bool:
        chang = True
    else:
        chang = False
    return chang, np.array(new_centroids)


def Kmeans(data,centroids):
    while True :
        change, new_centroids = assignment(data, centroids)
        print(change)
        if change == False:
           break
        else:
            centroids = new_centroids
    return new_centroids


if __name__ == '__main__':
    tweets=pd.read_json('tweets.json',dtype=str,lines=True)
    data = tweets[['text', 'id']]
    tweets = {}
    for i in range(len(data)):
        tweets.update({data.at[i,'id']:data.at[i,'text']})
    f=open('initial_centroids.txt', 'r').readlines()
    centroids = []
    for element in f:
        centroids.append(int(str.replace(element,',\n','')))
    new_centroids = Kmeans(tweets,centroids)
    np.savetxt('new_centroids.txt',new_centroids,fmt='%s')
