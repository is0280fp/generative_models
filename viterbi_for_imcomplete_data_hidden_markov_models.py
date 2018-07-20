# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:49:06 2018

@author: is028
"""

import numpy as np
import matplotlib.pyplot as plt
import hidden_markov_models
from imcomplete_data_hidden_markov_models import gaussian_pdf
from imcomplete_data_hidden_markov_models import gaussian_pdfs
from imcomplete_data_hidden_markov_models import compute_alpha_hat
from imcomplete_data_hidden_markov_models import compute_beta_hat
from imcomplete_data_hidden_markov_models import compute_xi


def viterbi(X, mean, var, pi, A):
    '''
    #  式(13.68)
    '''
    gaus_pdfs = gaussian_pdfs(X, mean, var)
    #  返り値: N×K
    N = gaus_pdfs.shape[0]
    K = gaus_pdfs.shape[1]
    prev_omega_n = np.log(pi) + gaus_pdfs[0]
    argmax_omega_lst = []
    #  foward
    for n in range(1, N):
        omega_n = np.log(gaus_pdfs[n]) + np.max(
            np.log(A)+prev_omega_n.reshape(K, 1), axis=0)
        argmax_omega_n = np.argmax(
                np.log(A)+prev_omega_n.reshape(K, 1), axis=0)
        argmax_omega_lst.append(argmax_omega_n)
        prev_omega_n = omega_n
    argmax_omega_lst = np.array(argmax_omega_lst)
    argmax_omega_N = np.argmax(omega_n)
    #  backward
    infered_z_lst = [argmax_omega_N]
    for n in range(N-1)[::-1]:
        argmax_num = infered_z_lst[-1]
        infered_z_lst.append(argmax_omega_lst[n, argmax_num])
    return np.array(infered_z_lst)[::-1]


if __name__ == '__main__':
    #  ハイパーパラメータ、ユーザが入力する
    K = 3
    max_iter = 1000
    tol = 1e-10

    #  サンプルデータ生成
    sampler = hidden_markov_models.GaussianHMM()
    Z, X = sampler(10000, complete_data=True)
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
        print("--------------------------------------------------------------")

    #  推定用サンプルを可視化
    print("real data")
    sampler.visualize(X)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    #  推定したパラメタモデルを使って新たなサンプルを生成
    z_1 = np.random.choice(K, p=pi)
    x_1 = np.random.normal(mean[z_1], std[z_1])
    z_new = [z_1]
    x_new = [x_1]
    for i in np.arange(1, N):
        z_new.append(np.random.choice(K, p=A[z_new[-1]]))
        x_new.append(np.random.normal(mean[z_new[-1]], std[z_new[-1]]))

    #  推定したパラメタモデルを使って生成したサンプルを可視化
    x_new = np.array(x_new)
    print("create data")
    sampler.visualize(x_new)
    plt.show()
    print("transition_matrix:", A)
    print("mean:", mean)
    print("std:", std)

    #  以下, viterbiアルゴリズムで推論
    Z_new, X_new = sampler(2000, complete_data=True)
    Z_hat = viterbi(X_new, mean, var, pi, A)
    #  PRMLのP. 150の識別不可能性のせいで
    #  Z_hatの状態番号とZの対応とZ_newの状態番号とZの対応がとれていない
    #  Z_hatの状態番号をZ_newの状態番号に対応される
    sort_num = np.argsort(mean)
    sort_num = np.argsort(sort_num)
    renumbered_Z_hat = sort_num[Z_hat]

    plt.title("compare to Z")
    plt.plot(Z_new, "b-", linewidth=2, alpha=1)
    plt.plot(renumbered_Z_hat, "r-", linewidth=2, alpha=0.5)
    plt.legend(["Z_new", "renumbered_Z_hat"])
    plt.show()
