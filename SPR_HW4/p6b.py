###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


if __name__ == '__main__':
    data = pd.read_csv('ALS_train.csv')
    data = data[['ALSFRS_Total_range','trunk_range','hands_range','ALSFRS_Total_min','leg_range','mouth_range','trunk_min','mouth_min'
                 ,'respiratory_range','hands_min']]
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    sse = {}
    for k in range(1, 15):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
        sse[k] = kmeans.inertia_
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()
