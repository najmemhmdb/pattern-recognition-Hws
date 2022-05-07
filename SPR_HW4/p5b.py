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


# initialization algorithm
def initialize_centroids(data, k):
    centroids = []
    ids = list(data.keys())
    centroids.append(ids[np.random.randint(0,251)])
    for c_id in range(k - 1):
        dist = []
        max_dis = 0
        max_id = 0
        for id in data.keys():
            a = data.get(str(id))
            min_v = 252
            for j in range(len(centroids)):
                temp_dist = jaccard_distance(str.split(a), str.split(data.get(str(centroids[j]))))
                min_v = min(min_v, temp_dist)
            dist.append(min_v)
            if min_v> max_dis:
                max_dis = min_v
                max_id = id
        next_centroid = max_id
        centroids.append(next_centroid)
    return centroids


if __name__ == '__main__':
    tweets = pd.read_json('tweets.json', dtype=str, lines=True)
    data = tweets[['text', 'id']]
    tweets = {}
    for i in range(len(data)):
        tweets.update({data.at[i, 'id']: data.at[i, 'text']})
    centroids = initialize_centroids(tweets, k=25)
    np.savetxt('centroids_b.txt', centroids, fmt='%s')

