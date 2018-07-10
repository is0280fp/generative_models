# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:30:17 2018
@author: yume
パターン認識と機械学習　下
p.154-155参照
"""


import numpy as np
import matplotlib.pyplot as plt
import hidden_markov_models


def gaussian_pdfs(x, mean, var):
    """
    式(9.23)
    """
    #  返り値: N×K次元のarray
    lkh = []
    for mean_k, var_k in zip(mean, var):
        lkh.append(np.exp(-(x-mean_k)**2/(2*var_k)) / (2*np.pi*var_k)**0.5)
    return np.array(lkh)


def gaussian_pdf(x, mean, var):
    """
    式(9.23)
    """
    #  返り値: N×K次元のarray
    lkh = []
    lkh.append(np.exp(-(x-mean)**2/(2*var)) / (2*np.pi*var)**0.5)
    return np.array(lkh)


def loglikelihoods(x, mean, var, A):
    """
    式(9.28)
    """
    #  返り値: スカラー
    lkh = []
    for j in np.arange(A.shape[0]):
        for k in np.arange(A.shape[1]):
            lkh.append(A[j, k] * gaussian_pdf(x, mean[k], var[k]))
    return np.sum(np.log(np.sum(lkh, 0)))


def loglikelihood(x, mean, var, A):
    """
    式(9.28)
    """
    #  返り値: スカラー
    lkh = A * gaussian_pdf(x, mean, var)
    return np.sum(np.log(np.sum(lkh, 0)))


def alpha(x, mean, var, init_alpha, A):
    #  返り値: 長さN, K次元のarray
    #  AをZn-1で周辺化, スカラーになる
    gaus_pdf = gaussian_pdfs(x, mean, var).transpose()
    alpha_lst = [init_alpha]
    for n in np.arange(len(x)-1):
        for j in np.arange(A.shape[0]):
            sum_lst = []
            for k in np.arange(A.shape[1]):
                cn = loglikelihood(x[0:n+1], mean[k], var[k], A[j, k])
                sum_lst.append(np.array(alpha_lst[-1])[k] * 1/cn * A[j, k])
            alpha_lst.append(
                    np.array(sum_lst).sum() * gaus_pdf[0:n+1, 0::].sum(axis=0))
    return np.array(alpha_lst)


def beta(x, mean, var, init_beta, A):
    #  返り値: 長さN, K次元のarray
    gaus_pdf = gaussian_pdfs(x, mean, var).transpose()
    beta_lst = [init_beta]
    for n in np.arange(len(x)-1)[::-1]:
        for j in np.arange(A.shape[0]):
            sum_lst = []
            for k in np.arange(A.shape[1]):
                cn = loglikelihood(x[n::], mean[k], var[k], A[j, k])
                sum_lst.append(np.array(beta_lst[-1])[k] * 1/cn * A[
                        j, k] * gaus_pdf[n::, 0::].sum(axis=0))
            beta_lst.append(np.array(sum_lst).sum(axis=0))
    return np.array(beta_lst)[::-1]


def guzai(x, mean, var, A, alpha, beta, gaus_pdf):
    #  返り値: K*Kのarray, 長さN
    #  cnは0~Nまで
    log_lkh = loglikelihoods(x, mean, var, A)
    gaus_pdf = gaus_pdf.transpose()
    guzai_lst = []
    a = alpha[:-1]
    b = beta[1:]
    for n in np.arange(len(x)-1):
        for j in np.arange(A.shape[0]):
            for k in np.arange(A.shape[1]):
                guzai_lst.append(
                        a[n, k] * gaus_pdf[n, k] * A[
                                j, k] * b[n, k] * 1/log_lkh)
    return np.array(guzai_lst).reshape(-1, K, K)


if __name__ == '__main__':
    #  ハイパーパラメータ、ユーザが入力する
    K = 3
    max_iter = 1000
    tol = 1e-5

    #  サンプルデータ生成
    sampler = hidden_markov_models.GaussianHMM()
    X = sampler(10000, complete_data=False)
    N = X.shape[0]

    #    パラメタ初期値設定
    #  式(13.18)
    A = np.array([[0.4, 0.3, 0.3]])
    mean = np.arange(1, K+1)
    var = np.arange(1, K+1)

    #    パラメタ初期値での対数尤度
    log_lkh_lst = []
    log_lkh = loglikelihoods(X, mean, var, A)
    log_lkh_lst.append(log_lkh)
    prev_log_lkh = - np.inf
    init_beta = np.ones(K)
    gaus_pdf = gaussian_pdfs(X, mean, var)

    for iteration in np.arange(max_iter):
#        assert prev_log_lkh < log_lkh
        prev_log_lkh = loglikelihoods(X, mean, var, A)
        init_alpha = (loglikelihoods(X, mean, var, A) * A).sum(axis=0)

        #  Eステップ(負担率の計算)
        #  gaus_pdf = p(Xn|Zn), shape(N, K)のarray
        gaus_pdf = gaussian_pdfs(X, mean, var)
        a = alpha(X, mean, var, init_alpha, A)
        b = beta(X, mean, var, init_beta, A)
        gammas = a * b
        guzai = guzai(X, mean, var, A, a, b, gaus_pdf)

        #  Mステップ(パラメタ値を再計算)
        #  式(9.27), スカラー
        Ns = gammas.sum(0)

        mean = []
        #  式(9.24), 式(13.20)
        for k in np.arange(K):
            mean.append((gammas[::, k] * X).sum() / Ns[k])
        mean = np.array(mean)

        var = []
        #  式(9.25), 式(13.21)
        for k in np.arange(K):
            var.append((
                    gammas[::, k] * (X - mean[k]) * (X - mean[k]).transpose()
                    ).sum() / Ns[k])
        var = np.array(var)

        #  式(13.19), K*Kのarray
        A = guzai.sum(0) / guzai.sum()

        #  対数尤度の計算
        log_lkh = loglikelihoods(X, mean, var, A)
        log_lkh_lst.append(log_lkh)

        #  収束判定
        diff = log_lkh - prev_log_lkh
        if diff < tol:
            break

        #  標準偏差の計算
        std = var ** 0.5

        #  対数尤度の出力
        print("prev log-likelihood", prev_log_lkh)
        print("log-lkielihood", log_lkh)
        print("mean", mean)
        print("var", var)
        print("std", std)
        print("A", A)

        #  各ガウス分布確率密度関数の描画
        x_scope = np.linspace(np.min(X), np.max(X), num=N)
        for k in np.arange(K):
            plt.plot(x_scope, gaussian_pdf(x_scope, mean[k], var[k]))
        plt.ylim(0, 0.5)
        plt.title("Components(weighted)")
        plt.show()

        #  混合ガウシアン分布の確率密度関数の描画
        pdfs = []
        for k in np.arange(K):
            pdfs.append(gaussian_pdf(x_scope, mean[k], var[k]))
        pdfs = np.array(pdfs)
        plt.plot(x_scope, pdfs.sum(axis=0))
        plt.ylim(0, 0.5)
        plt.title("pdf")
        plt.show()

        #  対数尤度のグラフ
        plt.plot(np.array(log_lkh_lst))
        plt.ylim(-45000, -35000)
        plt.title("log-likelihood")
        plt.grid()
        plt.show()
        print("diff", diff)
        print("-----------------------------------------------------------------------")

    print("real data")
    sampler.visualize(X)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    #  推定したパラメタモデルを使って新たなサンプルを生成
    z_new = np.random.choice(K, N, p=A)
    x_new = []
    for i in np.arange(N):
        k = z_new[i]
        x_new.append(np.random.normal(mean[k], std[k], 1))
    x_new = np.array(x_new)

    print("create data")
    sampler.visualize(x_new)
    print("probabilities", A)
    print("mean", mean)
    print("std", std)
