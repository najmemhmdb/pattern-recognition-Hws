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
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs)




if __name__ == '__main__':
    data = pd.read_csv('ALS_train.csv')
    data = data[['ALSFRS_Total_range','trunk_range','hands_range','ALSFRS_Total_min','leg_range','mouth_range','trunk_min','mouth_min'
                 ,'respiratory_range','hands_min']]
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    # model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,linkage='average')
    # model = model.fit(data)
    # predict = model.predict(data)
    # plt.title('Hierarchical Clustering Dendrogram Average Linkage')
    # plot_dendrogram(model, truncate_mode='level', p=3)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()

    k = 2
    agg = AgglomerativeClustering(n_clusters=2,linkage='single')
    y_predict = agg.fit_predict(data)
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    # centroids = agg.cluster_centers_
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
    # ax.set_title('Silhouette plot for the various clusters')
    plt.tight_layout()
    plt.suptitle(f' Silhouette analysis using k = {k}' + ' single dis. ', fontsize=16, fontweight='semibold')
    plt.show()


