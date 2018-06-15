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

    count0_0 = 0
    count0_1 = 0
    count0_2 = 0
    count1_0 = 0
    count1_1 = 0
    count1_2 = 0
    count2_0 = 0
    count2_1 = 0
    count2_2 = 0
    prev_z = z[:-1]
    for i in np.arange(1, N):
        if prev_z[i-1] == 0:
            if z[i] == 0:
                count0_0 += 1
            elif z[i] == 1:
                count0_1 += 1
            else:
                count0_2 += 1
        elif prev_z[i-1] == 1:
            if z[i] == 0:
                count1_0 += 1
            elif z[i] == 1:
                count1_1 += 1
            else:
                count1_2 += 1
        else:
            if z[i] == 0:
                count2_0 += 1
            elif z[i] == 1:
                count2_1 += 1
            else:
                count2_2 += 1

    count0 = count0_0 + count0_1 + count0_2
    count1 = count1_0 + count1_1 + count1_2
    count2 = count2_0 + count2_1 + count2_2
    transition_matrix = []
    transition_matrix.append([count0_0/count0, count0_1/count0, count0_2/count0])
    transition_matrix.append([count1_0/count1, count1_1/count1, count1_2/count1])
    transition_matrix.append([count2_0/count2, count2_1/count2, count2_2/count2])

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
