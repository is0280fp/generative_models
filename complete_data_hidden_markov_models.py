# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:24:45 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
import hidden_markov_models


def gaussian_paras(x):
    mean = x.mean()
    var = x.var()
    std = x.std()
    return (mean, var, std)


if __name__ == '__main__':
    sampler = hidden_markov_models.GaussianHMM()
    z, x = sampler(10000, complete_data=True)
    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    #  パラメタ推定処理
    K = len(np.unique(z))  # 3
    N = len(z)
    p_initial_state = [0.4, 0.3, 0.3]

    print("K:", K)
    print("p_initial_state:", p_initial_state)

    a = np.linspace(0, K-1, K)
    b = np.linspace(0, K-1, K)
    a, b = np.meshgrid(a, b)
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    pre_and_now_zs_grid = np.hstack((a, b))
    prev_z = z[:-1]
    count_list = []

    for zs in pre_and_now_zs_grid:
      count = 0
      for i in np.arange(1, N):
        if all(zs == np.array([z[i], prev_z[i-1]])):
          count += 1
      count_list.append(count)

    count_list = np.array(count_list).reshape(K, K)
    transition_matrix = []
    for k in range(K):
      for i in range(K):
        transition_matrix.append(count_list[k, i]/count_list[k].sum())
    transition_matrix = np.array(transition_matrix).reshape(K, K)

    xs = []
    for k in range(K):
        x_zi = x[z == k]
        xs.append(x_zi)

    mean_lst = []
    var_lst = []
    std_lst = []

    for k in range(K):
        mean, var, std = gaussian_paras(np.array(xs[k], dtype=np.float64))
        mean_lst.append(mean)
        var_lst.append(var)
        std_lst.append(std)
    means = np.array(mean_lst)
    stds = np.array(std_lst)

#  以下、モデル生成処理
    z_1 = np.random.choice(K, 1, p=p_initial_state)
    x_1 = np.random.normal(means[z_1], stds[z_1], 1)
    z_new = [z_1]
    x_new = [x_1]
    for i in np.arange(1, N):
        z_new.append(np.random.choice(K, 1,
                                      p=transition_matrix[int(z_new[-1])]))
        x_new.append(np.random.normal(
                means[int(z_new[-1])], stds[int(z_new[-1])], 1))

    x_new = np.array(x_new)
    print("transition_matrix:", transition_matrix)
    print("mean:", mean_lst)
    print("std:", std_lst)
    sampler.visualize(x_new)
