# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_clustering_data import generate_clustering_data


def distance_data(x, y):
    each_dis = np.linalg.norm(x - y, axis=1)
    return each_dis


def decide_cluster(each_dis, min_dis):
    for i in np.arange(len(each_dis)):
        if each_dis[i] == min_dis[i]:
            return i


def centroid(x):
    return x.mean(0)


if __name__ == '__main__':
    K = 4
    X = generate_clustering_data()
    N, D = X.shape
    indicas = np.random.choice(N, K, replace=False)
    #    (K, D)
    #  現在のクラスタリング重心
    centroids = X[indicas]
    cluster_lst = []

    #  クラスタリング
    #  一回目のクラスタリング
    for x in X:
        each_dis = distance_data(x, centroids)
        min_dis = each_dis.min()
        cluster_lst.append(decide_cluster(
                each_dis, np.full_like(each_dis, min_dis)))

    #  二回目以降のクラスタリング
    j = 0
    #  １ステップ前のクラスタリング重心
    mat = np.zeros((4, 2))
    mat[:] = np.nan
    prev_centroids = mat
    #  1ステップ前と現在のクラスタリング重心を比較、同じであればクラスタリング終了
    while np.array_equal(centroids, prev_centroids) is False:
        prev_centroids = centroids

        #  各クラスタの重心が求める
        centroids_lst = []
        for k in np.arange(K):
            indicas = np.array(np.where(np.array(
                    cluster_lst[N*j:]) == k))[0, :]
            cluster_xs = []
            for i in indicas:
                cluster_xs.append(X[i])
            centroids_lst.append(centroid(np.array(cluster_xs)))
        centroids = np.array(centroids_lst)

        #  重心を基にデータをクラスタリングしなおす
        for x in X:
            each_dis = distance_data(x, centroids)
            min_dis = each_dis.min()
            cluster_lst.append(decide_cluster(
                    each_dis, np.full_like(each_dis, min_dis)))

        #  ステップ数の更新
        j += 1
        #  クラスタリング重心の表示
        print("prev centroids", prev_centroids)
        print("now centroids", centroids)
        print("--------------------------------------------------------------")

    #  クラスタリング結果描画
    for k in np.arange(K):
        indicas = np.array(np.where(np.array(cluster_lst[-N:]) == k))[0, :]
        plt.plot(X[indicas][:, 0], X[indicas][:, 1], ".")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.show()
