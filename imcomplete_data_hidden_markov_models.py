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
    return np.array(lkh).T


def compute_alpha_hat(x, mean, var, pi, A, gaus_pdfs):
    #  返り値alpha: 長さN, K次元のarray
    #  返り値c: 長さN-1, 1次元のarray
    init_alpha = gaussian_pdfs(x[0], mean, var) * pi
    init_alpha_hat = init_alpha / init_alpha.sum()
    alpha_hat_lst = [init_alpha_hat]
    c_lst = [init_alpha.sum()]
    for n in range(1, N):
        alpha_n = (alpha_hat_lst[-1] * A.T).sum(1) * gaus_pdfs[n]
        cn = alpha_n.sum()
        alpha_hat_n = np.array(alpha_n / cn)
        alpha_hat_lst.append(alpha_hat_n)
        c_lst.append(cn)
    return np.array(alpha_hat_lst), np.array(c_lst)


def compute_beta_hat(A, c, gaus_pdfs):
    #  返り値: 長さN, K次元のarray
    init_beta = np.ones(K)
    beta_hat_lst = [init_beta]
    for n in range(N-1)[::-1]:
        beta_n = (beta_hat_lst[-1] * A * gaus_pdfs[n+1]).sum(1)
        beta_hat_n = np.array(beta_n / c[n+1])
        beta_hat_lst.append(beta_hat_n)
    return np.array(beta_hat_lst)[::-1]


def compute_xi(A, alpha, beta, gaus_pdf, c):
    #  返り値: K*Kのarray, 長さN-1
    K = A.shape[1]
    xi_lst = []
    alpha = alpha[:-1]
    beta = beta[1:]
    gaus_pdf = gaus_pdf[1:]
    c = c[1:]
    for n in range(N-1):
        xi_n = (np.ones((K, K))
                * alpha[n]).T * gaus_pdf[n] * A * beta[n] / c[n]
        xi_lst.append(xi_n)
    return np.array(xi_lst).reshape(-1, K, K)


if __name__ == '__main__':
    #  ハイパーパラメータ、ユーザが入力する
    np.random.seed(0)
    K = 3
    max_iter = 1000
    tol = 1e-10

    #  サンプルデータ生成
    sampler = hidden_markov_models.GaussianHMM()
    X = sampler(10000, complete_data=False)
    N = X.shape[0]

    #    パラメタ初期値設定
    pi = np.array([0.4, 0.3, 0.3])
    #  式(13.18)
    A = np.array([[0.4, 0.3, 0.3],
                  [0.3, 0.4, 0.3],
                  [0.3, 0.3, 0.4]
                  ])
    mean = np.array([1, 5, 10])
    var = np.array([1, 5, 10])

    #    パラメタ初期値での対数尤度
    log_lkh_lst = []
    log_lkh = -np.inf
    log_lkh_lst.append(log_lkh)

    for iteration in np.arange(max_iter):
        #  Eステップ(負担率の計算)
        #  gaus_pdf = p(Xn|Zn), shape(N, K)のarray
        gaus_pdfs = gaussian_pdfs(X, mean, var)
        alpha_hat, c = compute_alpha_hat(X, mean, var, pi, A, gaus_pdfs)
        beta_hat = compute_beta_hat(A, c, gaus_pdfs)
        gammas = alpha_hat * beta_hat
        xis = compute_xi(A, alpha_hat, beta_hat, gaus_pdfs, c)

        #  Mステップ(パラメタ値を再計算)
        #  式(9.27), スカラー
        Ns = gammas.sum(0)

        mean = []
        #  式(9.24), 式(13.20)
        for k in range(K):
            mean_k = (gammas[:, k] * X).sum() / Ns[k]
            mean.append(mean_k)
        mean = np.array(mean)

        var = []
        #  式(9.25), 式(13.21)
        for k in range(K):
            var_k = (gammas[:, k]
                     * (X - mean[k]) * (X - mean[k]).T).sum() / Ns[k]
            var.append(var_k)
        var = np.array(var)

        #  式(13.19), K*Kのarray
        sum_xis = xis.sum(0)
        A = sum_xis / sum_xis.sum(1, keepdims=True)

        #  対数尤度の計算
        prev_log_lkh = log_lkh
        log_lkh = np.sum(np.log(c))
        log_lkh_lst.append(log_lkh)
        assert prev_log_lkh < log_lkh

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
        plt.plot(x_scope, pdfs.sum(0))
        plt.ylim(0, 0.5)
        plt.title("pdf")
        plt.show()

        #  対数尤度のグラフ
        plt.plot(np.array(log_lkh_lst))
        plt.ylim(-80000, -18000)
        plt.title("log-likelihood")
        plt.grid()
        plt.show()
        print("diff", diff)
        print("-----------------------------------------------------------------------")

    #  推定用サンプルを可視化
    print("real data")
    sampler.visualize(X)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    #  推定したパラメタモデルを使って新たなサンプルを生成
    z_1 = np.random.choice(K, 1, p=pi)
    x_1 = np.random.normal(mean[z_1], std[z_1], 1)
    z_new = [z_1]
    x_new = [x_1]
    for i in np.arange(1, N):
        z_new.append(np.random.choice(K, 1, p=A[int(z_new[-1])]))
        x_new.append(np.random.normal(
                mean[int(z_new[-1])], std[int(z_new[-1])], 1))

    #  推定したパラメタモデルを使って生成したサンプルを可視化
    x_new = np.array(x_new)
    print("create data")
    sampler.visualize(x_new)
    print("transition_matrix:", A)
    print("mean:", mean)
    print("std:", std)
