# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import mixture_distributions


def likehood_function(x, mean_k, sigma_k, pi_k):
    lkh_lst = []
    for mean, sigma, pi in zip(mean_k, sigma_k, pi_k):
        lkh_lst.append(np.exp(-(X-mean)**2/2*sigma) / (2*np.pi*sigma)**0.5)
    return np.array(lkh_lst, dtype=np.float64)


def hist(data):
    x_min, x_max = int(min(data)), int(max(data)+1)
    bins = np.arange(x_min, x_max, 0.2)
    plt.hist(data, bins=bins, range=(x_min, x_max))


def calc_prob_gmm(data, mu, sigma, pi, K):
    return [[pi[k]*st.multivariate_normal.pdf(d, mu[k], sigma[k]) for k in range(K)] for d in data]


def print_gmm_contour(x, mu, sigma, pi, K):
    # display predicted scores by the model as a contour plot
    X, Y = np.meshgrid(np.linspace(x[:, 0].min(), x[:, 0].max()),
                       np.linspace(x[:, 1].min(), x[:, 1].max()))
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = np.sum(np.asanyarray(calc_prob_gmm(XX, mu, sigma, pi, K)), axis=1)
    Z = Z.reshape(X.shape)
    plt.contour(X, Y, Z, alpha=0.2, zorder=-100)
    plt.title('pdf contour of a GMM')


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
    log_lkh = np.log(lkh_k.sum(axis=0)).sum(axis=0)
    prev_log_lkh = np.log(lkh_k.sum(axis=0)).sum(axis=0)

    while True:
        prev_log_lkh = np.log(lkh_k.sum(axis=0)).sum(axis=0)

        #  Eステップ(負担率の計算)
        ganma_lst = []
        lkhs = lkh_k.sum(axis=0)
        for i in np.arange(K):
            ganma_lst.append(lkh_k[i]/lkhs)

        #  Mステップ(パラメタ値を再計算)
        N_k_lst = []
        for i in np.arange(K):
            N_k_lst.append(ganma_lst[i].sum(axis=0))
        N_k = np.array(N_k_lst)

        mean_k_lst = []
        for i in np.arange(K):
            mean_k_lst.append((ganma_lst[i] * X).sum(axis=0) / N_k[i])
        mean_k = np.array(mean_k_lst)

        sigma_k_lst = []
        for i in np.arange(K):
            sigma_k_lst.append((
                    ganma_lst[i] * (X - mean_k[i]) * (
                            X - mean_k[i]).transpose()).sum(axis=0) / N_k[i])
        sigma_k = np.array(sigma_k_lst)

        pi_k = N_k / N_k.sum(axis=0)
        lkh_k = likehood_function(X, mean_k, sigma_k, pi_k)

        #  対数尤度の計算
        log_lkh = np.log(lkh_k.sum(axis=0)).sum(axis=0)

        #  対数尤度の出力
        print("prev log-like-hood", prev_log_lkh)
        print("log-lkie-hood", log_lkh)
        print("mean_k", mean_k)
        print("sigma_k", sigma_k)
        print("pi_k", pi_k)

        #  各ガウス分布の描画
        std_k = sigma_k ** 0.5
        for k in np.arange(K):
            data = np.random.normal(
                    mean_k[k], std_k[k],
                    np.int64(np.round(pi_k[k]*np.float64(N))))
            hist(data)
        plt.title("each like-hood")
        plt.show()

        #  負担率を表現したデータ
        print_gmm_contour(X, mean_k, sigma_k, pi_k, K)
        print("-----------------------------------------------------------------------")
