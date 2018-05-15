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


def gamma(x, mean, var, pi):
    """
    式(9.23)
    """
    lkh = []
    for mean_k, var_k, pi_k in zip(mean, var, pi):
        lkh.append(pi_k * gaussian_pdf(x, mean_k, var_k))
    lkh = np.array(lkh)
    lkhs = np.sum(lkh, 0)
    return lkh/lkhs


def loglikelihood(x, mean, var, pi):
    """
    式(9.28)
    """
    lkh = []
    for mean_k, var_k, pi_k in zip(mean, var, pi):
        lkh.append(pi_k * gaussian_pdf(x, mean_k, var_k))
    return np.sum(np.log(np.sum(lkh, 0)))


if __name__ == '__main__':
    sampler = mixture_distributions.MixtureOfGaussians()
    X = sampler(1000, complete_data=False)
#    sampler.visualize(X)
#    print("Distribution: ", sampler.get_name())
#    print("Parameters: ", sampler.get_params())

    K = 3
    N = X.shape[0]
    #    パラメタ初期値設定
    mean = np.arange(1, K+1)
    var = np.arange(1, K+1)
    pi = np.random.dirichlet([1] * K)
    #    パラメタ初期値での対数尤度
    log_lkh_lst = []
    log_lkh = loglikelihood(X, mean, var, pi)
    log_lkh_lst.append(log_lkh)
    prev_log_lkh = loglikelihood(X, mean, var, pi)

    while True:
#        assert prev_log_lkh < log_lkh
        prev_log_lkh = loglikelihood(X, mean, var, pi)

        #  Eステップ(負担率の計算)
        gammas = gamma(X, mean, var, pi)

        #  Mステップ(パラメタ値を再計算)
        #  式(9.27)
        Ns = []
        for k in np.arange(K):
            Ns.append(gammas[k].sum())
        Ns = np.array(Ns)

        mean = []
        #  式(9.24)
        for gamma_k, N_k in zip(gammas, Ns):
            mean.append((gamma_k * X).sum() / N_k)
        mean = np.array(mean)

        var = []
        #  式(9.25)
        for gamma_k, N_k, mean_k in zip(gammas, Ns, mean):
            var.append((
                    gamma_k * (X - mean_k) * (X - mean_k).transpose()
                    ).sum() / N_k)
        var = np.array(var)

        #  式(9.26)
        pi = Ns / Ns.sum()

        #  対数尤度の計算
        log_lkh = loglikelihood(X, mean, var, pi)
        log_lkh_lst.append(log_lkh)

        #  標準偏差の計算
        std = var ** 0.5

        #  対数尤度の出力
        print("prev log-likelihood", prev_log_lkh)
        print("log-lkielihood", log_lkh)
        print("mean", mean)
        print("var", var)
        print("std", std)
        print("pi", pi)

        #  各ガウス分布確率密度関数の描画
        x_scope = np.linspace(np.min(X), np.max(X), num=N)
        for k in np.arange(K):
            plt.plot(x_scope, gaussian_pdf(x_scope, mean[k], var[k]))
        plt.ylim(0, 0.5)
        plt.title("pdfs")
        plt.show()

        #  混合ガウシアン分布の確率密度関数の描画
        pdfs = []
        for k in np.arange(K):
            pdfs.append(gaussian_pdf(x_scope, mean[k], var[k]))
        pdfs = np.array(pdfs)
        plt.plot(x_scope, pdfs.sum(axis=0))
        plt.ylim(0, 0.5)
        plt.title("stacked")
        plt.show()

        #  対数尤度のグラフ
        plt.plot(np.array(log_lkh_lst))
        plt.ylim(-45000, -2000)
        plt.title("log-likelihood")
        plt.grid()
        plt.show()
        print("-----------------------------------------------------------------------")
