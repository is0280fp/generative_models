# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018
@author: yume
パターン認識と機械学習　下
p.154-155参照
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import mixture_distributions


def gaussian_pdf(x, mean, var):
    return np.exp(-(x-mean)**2/(2*var)) / (2*np.pi*var)**0.5


def gamma(x, mean_k, var_k, pi_k):
    """
    式(9.23)
    """
    lkh = []
    for mean, var, pi in zip(mean_k, var_k, pi_k):
        lkh.append(pi * gaussian_pdf(x, mean, var))
    lkh = np.array(lkh)
    lkhs = np.sum(lkh, 0)
    return lkh/lkhs


def loglikelihood(x, mean_k, var_k, pi_k):
    """
    式(9.28)
    """
    lkh_lst = []
    for mean, var, pi in zip(mean_k, var_k, pi_k):
        lkh_lst.append(pi * gaussian_pdf(x, mean, var))
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
        gammas = gamma(X, mean_k, var_k, pi_k)

        #  Mステップ(パラメタ値を再計算)
        #  式(9.27)
        N_k = []
        for k in np.arange(K):
            N_k.append(gammas[k].sum())
        N_k = np.array(N_k)

        mean_k = []
        #  式(9.24)
        for gamma_k, n_k in zip(gammas, N_k):
            mean_k.append((gamma_k * X).sum() / n_k)
        mean_k = np.array(mean_k)

        var_k = []
        #  式(9.25)
        for gamma_k, n_k, mean in zip(gammas, N_k, mean_k):
            var_k.append((
                    gamma_k * (X - mean) * (X - mean).transpose()).sum() / n_k)
        var_k = np.array(var_k)

        #  式(9.26)
        pi_k = N_k / N_k.sum()

        #  対数尤度の計算
        log_lkh = loglikelihood(X, mean_k, var_k, pi_k)

        #  標準偏差の計算
        std_k = var_k ** 0.5

        #  対数尤度の出力
        print("prev log-likelihood", prev_log_lkh)
        print("log-lkielihood", log_lkh)
        print("mean_k", mean_k)
        print("var_k", var_k)
        print("std_k", std_k)
        print("pi_k", pi_k)

        #  各ガウス分布確率密度関数の描画
        x_scope = np.linspace(np.min(X), np.max(X), num=N)
        for k in np.arange(K):
            plt.plot(x_scope, gaussian_pdf(x_scope, mean_k[k], var_k[k]))
        plt.ylim(0, 0.5)
        plt.title("pdfs")
        plt.show()

        #  混合ガウシアン分布の確率密度関数の描画
        pdfs = []
        for k in np.arange(K):
            pdfs.append(gaussian_pdf(x_scope, mean_k[k], var_k[k]))
        pdfs = np.array(pdfs)
        plt.plot(x_scope, pdfs.sum(axis=0))
        plt.ylim(0, 0.5)
        plt.title("stacked")
        plt.show()

        #  対数尤度のグラフ

        print("-----------------------------------------------------------------------")
