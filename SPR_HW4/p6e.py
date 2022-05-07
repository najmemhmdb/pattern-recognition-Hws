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
from sklearn.metrics import silhouette_samples, silhouette_score



if __name__ == '__main__':
    data = pd.read_csv('ALS_train.csv')
    data = data[['ALSFRS_Total_range','trunk_range','hands_range','ALSFRS_Total_min','leg_range','mouth_range','trunk_min','mouth_min'
                 ,'respiratory_range','hands_min']]
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    k = 2
    km = KMeans(n_clusters=k,init='k-means++')
    y_predict = km.fit_predict(data)
# silhouette
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    centroids = km.cluster_centers_
    silhouette_vals = silhouette_samples(data, y_predict)
    y_ticks = []
    y_lower = y_upper = 0
    for i, cluster in enumerate(np.unique(y_predict)):
        cluster_silhouette_vals = silhouette_vals[y_predict == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)

        ax.barh(range(y_lower, y_upper),
                cluster_silhouette_vals, height=1);
        ax.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)
        avg_score = np.mean(silhouette_vals)
        ax.axvline(avg_score, linestyle='--',
                   linewidth=2, color='green')
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.set_xlabel('Silhouette coefficient values')
    ax.set_ylabel('Cluster labels')
    ax.set_title('Silhouette plot for the various clusters')
    plt.tight_layout()
    plt.suptitle(f' Silhouette analysis using k = {k}', fontsize=16, fontweight='semibold')
    plt.show()



# bar plot
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    width = 0.3
    N = 2
    arr = np.zeros([1,N])
    for i in range(N):
        length = len(([data[k] for k in range(len(data)) if y_predict[k] == i ]))
        arr[0,i] = length
        plt.text(i,length, str(length/len(y_predict) * 100)[0:4] + '%')
    d = tuple(map(tuple, arr))
    ind = np.arange(N)
    ax.bar(ind, d[0], width, color='lightgreen')
    plt.show()