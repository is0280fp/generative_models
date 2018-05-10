# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import mixture_distributions


def likehood_function(x, mean_k, var_k, pi_k):
    lkh_lst = []
    for mean, var, pi in zip(mean_k, var_k, pi_k):
        lkh_lst.append(np.exp(-(X-mean)**2/2*var) / (2*np.pi*var)**0.5)
    return np.array(lkh_lst, dtype=np.float64)


def hist(data):
    x_min, x_max = int(min(data)), int(max(data)+1)
    bins = np.arange(x_min, x_max, 0.2)
    plt.hist(data, bins=bins, range=(x_min, x_max))


if __name__ == '__main__':
    sampler = mixture_distributions.MixtureOfGaussians()
    X = sampler(1000, complete_data=False)
#    sampler.visualize(X)
#    print("Distribution: ", sampler.get_name())
#    print("Parameters: ", sampler.get_params())

    K = 3
    N = X.shape[0]
    #    パラメタ初期値設定
    mean_k = np.arange(1, K+1)
    var_k = np.arange(1, K+1)
    pi_k = np.random.dirichlet([1] * K)
    #    パラメタ初期値での対数尤度
    lkh_k = likehood_function(X, mean_k, var_k, pi_k)
    log_lkh = np.log(lkh_k.sum(axis=0)).sum()
    prev_log_lkh = np.log(lkh_k.sum(axis=0)).sum()

    while True:
        assert prev_log_lkh <= log_lkh
        prev_log_lkh = np.log(lkh_k.sum(axis=0)).sum()

        #  Eステップ(負担率の計算)
        ganma_lst = []
        lkhs = lkh_k.sum(axis=0)
        for i in np.arange(K):
            ganma_lst.append(lkh_k[i]/lkhs)

        #  Mステップ(パラメタ値を再計算)
        N_k_lst = []
        for i in np.arange(K):
            N_k_lst.append(ganma_lst[i].sum())
        N_k = np.array(N_k_lst)

        mean_k_lst = []
        for i in np.arange(K):
            mean_k_lst.append((ganma_lst[i] * X).sum() / N_k[i])
        mean_k = np.array(mean_k_lst)

        var_k_lst = []
        for i in np.arange(K):
            var_k_lst.append((
                    ganma_lst[i] * (X - mean_k[i]) * (
                            X - mean_k[i]).transpose()).sum() / N_k[i])
        var_k = np.array(var_k_lst)

        pi_k = N_k / N_k.sum()
        lkh_k = likehood_function(X, mean_k, var_k, pi_k)

        #  対数尤度の計算
        log_lkh = np.log(lkh_k.sum(axis=0)).sum()

        #  対数尤度の出力
        print("prev log-like-hood", prev_log_lkh)
        print("log-lkie-hood", log_lkh)
        print("mean_k", mean_k)
        print("var_k", var_k)
        print("pi_k", pi_k)

        #  各ガウス分布の描画
        std_k = var_k ** 0.5
        for k in np.arange(K):
            data = np.random.normal(mean_k[k], std_k[k], N)
            hist(data)
        plt.title("each like-hood")
        plt.show()

        #  負担率を表現したデータ
#        ganmas = np.array(ganma_lst)
#        for i in np.arange(N):
#            plt.bar(X[i], ganmas[:, i][0], color='b')
#            plt.bar(X[i], ganmas[:, i][1], color='g')
#            plt.bar(X[i], ganmas[:, i][2], color='r')
#        plt.xlim(X.min(), X.max())
#        plt.show()
        print("-----------------------------------------------------------------------")
