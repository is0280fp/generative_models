# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from generate_clustering_data import generate_clustering_data


if __name__ == '__main__':
    K = 4
    X = generate_clustering_data()
    N, D = X.shape
    indices = np.random.choice(N, K, replace=False)
    #    (K, D)
    #  クラスタリング重心の初期値
    centroids = X[indices]

    #  クラスタリング
    each_dis = pairwise_distances(X, centroids)
    clusters = each_dis.argmin(axis=1)

    #  初期値に対する１ステップ前のクラスタリング重心
    prev_centroids = np.full_like(centroids, np.nan)
    #  1ステップ前と現在のクラスタリング重心を比較、同じであればクラスタリング終了
    while np.array_equal(centroids, prev_centroids) is False:
        #  １ステップ前のクラスタリング重心
        prev_centroids = centroids

        #  各クラスタの重心が求める
        centroids_lst = []
        for k in np.arange(K):
            cluster_xs = X[clusters == k]
            centroids_lst.append(cluster_xs.mean(axis=0))
        centroids = np.array(centroids_lst)

        #  重心を基にデータをクラスタリングしなおす
        each_dis = pairwise_distances(X, centroids)
        clusters = each_dis.argmin(axis=1)

        #  クラスタリング重心の表示
        print("prev centroids", prev_centroids)
        print("now centroids", centroids)
        print("--------------------------------------------------------------")

    #  クラスタリング結果描画
    for k in np.arange(K):
        plt.plot(X[clusters == k][:, 0], X[clusters == k][:, 1], ".")
        plt.plot(centroids[0:, 0], centroids[0:, 1], "*")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.show()
