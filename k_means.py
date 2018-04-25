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
#  現在のクラスタリング重心
    now_centroids = centroids_lst[-K:]
#  １ステップ前のクラスタリング重心
    mat = np.zeros(2)
    mat[:] = np.nan
    initial_lst = []
    for i in np.arange(K):
        initial_lst.append(mat)
    prev_centroids = initial_lst
#  1ステップ前と現在のクラスタリング重心を比較、同じであればクラスタリング終了
    while np.array_equal(now_centroids, prev_centroids) is False:
        #  各クラスタの重心が求める
        for k in np.arange(K):
            indicas = np.array(np.where(np.array(
                    cluster_lst[N*j:]) == k))[0, :]
            cluster_xs = []
            for i in indicas:
                cluster_xs.append(X[i])
            centroids_lst.append(centroid(np.array(cluster_xs)))
        #  重心を基にデータをクラスタリングしなおす
        for x in X:
            each_dis = distance_data(np.full_like(
                    np.array(centroids_lst[-K:]), x), np.array(
                            centroids_lst[-K:]))
            min_dis = each_dis.min()
            cluster_lst.append(decide_cluster(
                    each_dis, np.full_like(each_dis, min_dis)))

        #  ステップ数とクラスタリング重心の更新
        j += 1
        now_centroids = centroids_lst[-K:]
        prev_centroids = centroids_lst[-K*2:-K]
        #  クラスタリング重心の表示
        print("prev centroids", prev_centroids)
        print("now centroids", now_centroids)
        print("--------------------------------------------------------------")

#  クラスタリング結果描画
    for k in np.arange(K):
        indicas = np.array(np.where(np.array(cluster_lst[-N:]) == k))[0, :]
        plt.plot(X[indicas][:, 0], X[indicas][:, 1], ".")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.show()
