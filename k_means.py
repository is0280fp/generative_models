# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_clustering_data import generate_clustering_data


def distance_data(X, centroids):
    each_dis = []
    for x in X:
        each_dis.append(np.linalg.norm(x - centroids, axis=1))
    return np.array(each_dis)


if __name__ == '__main__':
    K = 4
    X = generate_clustering_data()
    N, D = X.shape
    indicas = np.random.choice(N, K, replace=False)
    #    (K, D)
    #  現在のクラスタリング重心
    centroids = X[indicas]

    #  クラスタリング
    #  一回目のクラスタリング
    each_dis = distance_data(X, centroids)
    clusters = each_dis.argmin(axis=1)

    #  二回目以降のクラスタリング
    j = 0
    #  １ステップ前のクラスタリング重心
    prev_centroids = np.zeros((K, D))
    prev_centroids[:] = np.nan
    #  1ステップ前と現在のクラスタリング重心を比較、同じであればクラスタリング終了
    while np.array_equal(centroids, prev_centroids) is False:
        prev_centroids = centroids

        #  各クラスタの重心が求める
        centroids_lst = []
        for k in np.arange(K):
            cluster_xs = X[clusters == k]
            centroids_lst.append(cluster_xs.mean(axis=0))
        centroids = np.array(centroids_lst)

        #  重心を基にデータをクラスタリングしなおす
        each_dis = distance_data(X, centroids)
        clusters = each_dis.argmin(axis=1)

        #  ステップ数の更新
        j += 1
        #  クラスタリング重心の表示
        print("prev centroids", prev_centroids)
        print("now centroids", centroids)
        print("--------------------------------------------------------------")

    #  クラスタリング結果描画
    for k in np.arange(K):
        plt.plot(X[clusters == k][:, 0], X[clusters == k][:, 1], ".")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.show()
