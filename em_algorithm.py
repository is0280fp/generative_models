# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
import mixture_distributions


def likehood_function(x, mean_k, sigma_k, pi_k):
    lkh_lst = []
    for mean, sigma, pi in zip(mean_k, sigma_k, pi_k):
        lkh_lst.append(np.exp(-(X-mean)**2/2*sigma) / (2*np.pi*sigma)**0.5)
    return np.array(lkh_lst, dtype=np.float64)


if __name__ == '__main__':
    sampler = mixture_distributions.MixtureOfGaussians()
    X = sampler(10000, complete_data=False)
#    sampler.visualize(X)
#    print("Distribution: ", sampler.get_name())
#    print("Parameters: ", sampler.get_params())

    K = 3
    N = X.shape
    #    パラメタ初期値設定
    mean_k = np.arange(1, K+1)
    sigma_k = np.arange(1, K+1)
    pi_k = np.arange(1, K+1)
    #    パラメタ初期値での対数尤度
    lkh_k = likehood_function(X, mean_k, sigma_k, pi_k)
    log_lkh = np.log(lkh_k.sum()).sum()
    prev_log_lkh = np.log(lkh_k.sum()).sum()

    while True:
        prev_log_lkh = np.log(lkh_k.sum()).sum()

        #  Eステップ(負担率の計算)
        ganma_lst = []
        lkhs = lkh_k.sum()
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

        sigma_k_lst = []
        for i in np.arange(K):
            sigma_k_lst.append((
                    ganma_lst[i] * (X - mean_k[i]) * (
                            X - mean_k[i]).transpose()).sum() / N_k[i])
        sigma_k = np.array(sigma_k_lst)

        pi_k = N_k / N_k.sum()
        lkh_k = likehood_function(X, mean_k, sigma_k, pi_k)

        #  対数尤度の計算
        log_lkh = np.log(lkh_k.sum()).sum()

        #  対数尤度の出力
        print("prev log-like-hood", prev_log_lkh)
        print("log-lkie-hood", log_lkh)
        print("mean_k", mean_k)
        print("sigma_k", sigma_k)
        print("pi_k", pi_k)
        print("-----------------------------------------------------------------------")
