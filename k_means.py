# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_clustering_data import generate_clustering_data


def distance_data_mean(x, mean):
    return np.linalg.norm(x-mean)


if __name__ == '__main__':
    K = 4
    X = generate_clustering_data()
    sets_cluster = []
    cluster_mean_points_lst = []
    prev_cluster_mean_points = np.zeros((K, 2), dtype=int)

    plt.plot(X[:, 0], X[:, 1], '.')
    plt.title('Number of clusters is 4.')
    plt.show()

    N = np.arange(len(X))
    y = np.random.choice(N, K)
    for i in y:
        cluster_mean_points_lst.append(X[i])
    cluster_mean_points = X[y]

    while (prev_cluster_mean_points != cluster_mean_points).all():
        for x in X:
            dis_lst = []
            for cluster_mean_point in cluster_mean_points:
                dis_lst.append(distance_data_mean(x, cluster_mean_point))
            cluster = np.argmin(np.array(dis_lst))
            sets_cluster.append(cluster)

        prev_cluster_mean_points = cluster_mean_points_lst[-K:]
        for k in np.arange(K):
            indexes = np.array(np.where(sets_cluster == k)).transpose()
            xs = []
            for i in indexes:
                xs.append(X[i])
            cluster_mean_points_lst.append(np.array(xs).mean(
                    axis=0, dtype=np.float64))
        cluster_mean_points = cluster_mean_points_lst[-K:]
