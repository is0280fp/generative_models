# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_clustering_data import generate_clustering_data


def distance_data(X, Y):
    each_dis = []
    for x, y in zip(X, Y):
        each_dis.append(np.linalg.norm(x - y))
    return np.array(each_dis)


def decide_cluster(each_dis, min_dis):
    for i in np.arange(len(each_dis)):
        if each_dis[i] == min_dis[i]:
            return i


def centroid(x):
    return np.array([x[:, 0].mean(), x[:, 1].mean()])


if __name__ == '__main__':
    K = 4
    X = generate_clustering_data()


#    plt.plot(X[:, 0], X[:, 1], '.')
#    plt.title('Number of clusters is 4.')
#    plt.show()

    N, D = X.shape
    indicas = np.random.choice(N, K, replace=False)
#    (K, D)
    centroids = X[indicas]
    cluster_lst = []
    centroids_lst = []
    for x in centroids:
        centroids_lst.append(x)

#  クラスタリング
#  一回目のクラスタリング
    for x in X:
        each_dis = distance_data(np.full_like(centroids, x), centroids)
        min_dis = each_dis.min()
        cluster_lst.append(decide_cluster(
                each_dis, np.full_like(each_dis, min_dis)))

#  二回目以降のクラスタリング
    j = 0
    now_centroids = centroids_lst[-K:]
    mat = np.zeros(2)
    mat[:] = np.nan
    initial_lst = []
    for i in np.arange(K):
        initial_lst.append(mat)
    prev_centroids = initial_lst
    while np.array_equal(now_centroids, prev_centroids) is False:
        for k in np.arange(K):
            indicas = np.array(np.where(np.array(
                    cluster_lst[N*j:]) == k))[0, :]
            cluster_xs = []
            for i in indicas:
                cluster_xs.append(X[i])
            centroids_lst.append(centroid(np.array(cluster_xs)))
        #  ここまでで、各クラスタの重心が求まった
        for x in X:
            each_dis = distance_data(np.full_like(
                    np.array(centroids_lst[-K:]), x), np.array(
                            centroids_lst[-K:]))
            min_dis = each_dis.min()
            cluster_lst.append(decide_cluster(
                    each_dis, np.full_like(each_dis, min_dis)))

        j += 1
        now_centroids = centroids_lst[-K:]
        prev_centroids = centroids_lst[-K*2:-K]
        print("prev centroids", prev_centroids)
        print("now centroids", now_centroids)
        print("--------------------------------------------------------------")
