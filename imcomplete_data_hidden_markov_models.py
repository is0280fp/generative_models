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


def gaussian_pdf(x, mean, var):
    """
    式(9.23)
    """
    #  返り値: N×K次元のarray
    return np.exp(-(x-mean)**2/(2*var)) / (2*np.pi*var)**0.5


def gaussian_pdfs(x, mean, var):
    """
    式(9.23)
    """
    #  返り値: N×K次元のarray
    lkh = []
    for mean_k, var_k in zip(mean, var):
        lkh.append(gaussian_pdf(x, mean_k, var_k))
    return np.array(lkh).transpose()


def loglikelihoods(x, mean, var, A):
    """
    式(9.28)
    """
    #  返り値: スカラー
    lkh = []
    for j in range(A.shape[0]):
        for k in range(A.shape[1]):
            lkh.append(A[j, k] * gaussian_pdf(x, mean[k], var[k]))
    return np.sum(np.log(np.sum(lkh, 0)))


def compute_alpha_hat(x, mean, var, init_alpha, A):
    #  返り値alpha: 長さN, K次元のarray
    #  返り値c: 長さN-1, 1次元のarray
    gaus_pdf = gaussian_pdfs(x, mean, var)
    alpha_lst = []
    alpha_hat_lst = [init_alpha]
    c_lst = []
    J = A.shape[0]
    K = A.shape[1]
    for n in range(1, N):
        for k in range(K):
            sum_lst = []
            for j in range(J):
                sum_lst.append(np.array(alpha_hat_lst[-1])[k] * A[j, k])
            alpha_lst.append(np.array(sum_lst).sum())
        alpha_lst[-K::] = alpha_lst[-K::] * gaus_pdf[n]
        cn = np.array(alpha_lst[-K::]).sum()
        alpha_hat_lst.append(alpha_lst[-K::]/cn)
        c_lst.append(cn)
    return np.array(alpha_hat_lst), np.array(c_lst)


def compute_beta_hat(x, mean, var, init_beta, A, c):
    #  返り値: 長さN, K次元のarray
    gaus_pdf = gaussian_pdfs(x, mean, var)
    beta_lst = []
    beta_hat_lst = [init_beta]
    J = A.shape[0]
    K = A.shape[1]
    for n in range(N-1)[::-1]:
        for j in range(J):
            sum_lst = []
            for k in range(K):
                sum_lst.append(np.array(
                        beta_hat_lst[-1])[k] * A[j, k] * gaus_pdf[n, k])
            beta_lst.append(np.array(sum_lst).sum())
        beta_hat_lst.append(beta_lst[-K::]/c[n])
    return np.array(beta_hat_lst)[::-1]


def xi(x, mean, var, A, alpha, beta, gaus_pdf, c):
    #  返り値: K*Kのarray, 長さN-1
    xi_lst = []
    a = alpha[:-1]
    b = beta[1:]
    J = A.shape[0]
    K = A.shape[1]
    for n in range(N-1):
        for j in range(J):
            for k in range(K):
                xi_lst.append(
                        a[n, j] * gaus_pdf[n, k] * A[j, k] * b[n, k] * 1/c[n])
    return np.array(xi_lst).reshape(-1, K, K)


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
    A = np.array([[0.4, 0.3, 0.3],
                  [0.4, 0.3, 0.3],
                  [0.4, 0.3, 0.3]
                  ])
    mean = np.arange(1, K+1)
    var = np.arange(1, K+1)

    #    パラメタ初期値での対数尤度
    log_lkh_lst = []
    log_lkh = loglikelihoods(X, mean, var, A)
    log_lkh_lst.append(log_lkh)
    prev_log_lkh = - np.inf
    init_beta = np.ones(K)
    gaus_pdf = gaussian_pdfs(X, mean, var)
    pi = np.array([0.4, 0.3, 0.3])

    for iteration in np.arange(max_iter):
#        assert prev_log_lkh < log_lkh
        prev_log_lkh = loglikelihoods(X, mean, var, A)
        init_alpha = gaussian_pdfs(X[0], mean, var) * pi

        #  Eステップ(負担率の計算)
        #  gaus_pdf = p(Xn|Zn), shape(N, K)のarray
        gaus_pdf = gaussian_pdfs(X, mean, var)
        alpha_hat, c = compute_alpha_hat(X, mean, var, init_alpha, A)
        beta_hat = compute_beta_hat(X, mean, var, init_beta, A, c)
        gammas = alpha_hat * beta_hat
        xis = xi(X, mean, var, A, alpha_hat, beta_hat, gaus_pdf, c)

        #  Mステップ(パラメタ値を再計算)
        #  式(9.27), スカラー
        Ns = gammas.sum(0)

        mean = []
        #  式(9.24), 式(13.20)
        for k in range(K):
            mean.append((gammas[::, k] * X).sum() / Ns[k])
        mean = np.array(mean)

        var = []
        #  式(9.25), 式(13.21)
        for k in range(K):
            var.append((
                    gammas[::, k] * (X - mean[k]) * (X - mean[k]).transpose()
                    ).sum() / Ns[k])
        var = np.array(var)

        #  式(13.19), K*Kのarray
        A = xis.sum(0) / xis.sum()

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
        print("alpha_hat", alpha_hat)
        print("beta_hat", beta_hat)
        print("xi", xis)

        #  各ガウス分布確率密度関数の描画
        x_scope = np.linspace(np.min(X), np.max(X), num=N)
        for k in range(K):
            plt.plot(x_scope, gaussian_pdf(x_scope, mean[k], var[k]))
        plt.ylim(0, 0.5)
        plt.title("Components(weighted)")
        plt.show()

        #  混合ガウシアン分布の確率密度関数の描画
        pdfs = []
        for k in range(K):
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
    for i in range(N):
        k = z_new[i]
        x_new.append(np.random.normal(mean[k], std[k], 1))
    x_new = np.array(x_new)

    print("create data")
    sampler.visualize(x_new)
    print("probabilities", A)
    print("mean", mean)
    print("std", std)
