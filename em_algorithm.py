# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018
@author: yume
パターン認識と機械学習　下
p.154-155参照
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import mixture_distributions


def ganma(x, mean_k, var_k, pi_k):
    """
    式(9.23)
    """
    lkh = []
    for mean, var, pi in zip(mean_k, var_k, pi_k):
        lkh.append(pi * np.exp(-(x-mean)**2/(2*var)) / (2*np.pi*var)**0.5)
    lkh = np.array(lkh)
    lkhs = np.sum(lkh, 0)
    return lkh/lkhs


def loglikelihood(x, mean_k, var_k, pi_k):
    """
    式(9.28)
    """
    lkh_lst = []
    for mean, var, pi in zip(mean_k, var_k, pi_k):
        lkh_lst.append(pi * np.exp(-(x-mean)**2/(2*var)) / (2*np.pi*var)**0.5)
    return np.sum(np.log(np.sum(lkh_lst, 0)))


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
    log_lkh = loglikelihood(X, mean_k, var_k, pi_k)
    prev_log_lkh = loglikelihood(X, mean_k, var_k, pi_k)

    while True:
#        assert prev_log_lkh <= log_lkh
        prev_log_lkh = loglikelihood(X, mean_k, var_k, pi_k)

        #  Eステップ(負担率の計算)
        ganmas = ganma(X, mean_k, var_k, pi_k)

        #  Mステップ(パラメタ値を再計算)
        #  式(9.27)
        N_k = []
        for k in np.arange(K):
            N_k.append(ganmas[k].sum())
        N_k = np.array(N_k)

        mean_k = []
        #  式(9.24)
        for ganma_k, n_k in zip(ganmas, N_k):
            mean_k.append((ganma_k * X).sum() / n_k)
        mean_k = np.array(mean_k)

        var_k = []
        #  式(9.25)
        for ganma_k, n_k, mean in zip(ganmas, N_k, mean_k):
            var_k.append((
                    ganma_k * (X - mean) * (X - mean).transpose()).sum() / n_k)
        var_k = np.array(var_k)

        #  式(9.26)
        pi_k = N_k / N_k.sum()

        #  対数尤度の計算
        log_lkh = loglikelihood(X, mean_k, var_k, pi_k)

        #  対数尤度の出力
        print("prev log-like-hood", prev_log_lkh)
        print("log-lkie-hood", log_lkh)
        print("mean_k", mean_k)
        print("var_k", var_k)
        print("pi_k", pi_k)

        #  各ガウス分布の描画
        std_k = var_k ** 0.5
        z_new = np.random.choice(K, N, p=pi_k)
        z_counts = np.bincount(z_new)
        for k in np.arange(K):
            x_new = np.random.normal(mean_k[k], std_k[k], z_counts[k])
            hist(x_new)
        plt.title("each likelihood")
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
