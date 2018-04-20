# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:38:29 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import mixture_distributions


def gaussian_paras(x):
    mean = x.mean()
    var = x.var()
    std = x.std()
    return (mean, var, std)


if __name__ == '__main__':
    sampler = mixture_distributions.MixtureOfGaussians()
    z, x = sampler(10000, complete_data=True)
    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

#  以下、パラメタ推定処理
    K = len(np.unique(z))  # 3
    N = len(z)
    x_number = np.bincount(z)
    probabilities = np.array(x_number/N)

    print("K:", K)
    print("p:", probabilities)

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

#  以下、モデル生成処理
    Z = np.random.choice(K, N, p=probabilities)

    y0_lst = []
    y1_lst = []
    y2_lst = []
    for i in np.arange(len(Z)):
        if z[i] == 0:
            y0_lst.append([i, np.random.normal(mean_lst[0], std_lst[0], 1)])
        elif z[i] == 1:
            y1_lst.append([i, np.random.normal(mean_lst[1], std_lst[1], 1)])
        else:
            y2_lst.append([i, np.random.normal(mean_lst[2], std_lst[2], 1)])

    y_lst = []
    for y in y0_lst:
        y_lst.append(y)

    for y in y1_lst:
        y_lst.append(y)

    for y in y2_lst:
        y_lst.append(y)

    y = np.array(y_lst, dtype=np.float64)
    y = sorted(y, key=itemgetter(0))
    print("probabilities", probabilities)
    print("mean", mean_lst)
    print("var", var_lst)
    print("std", std_lst)
    sampler.visualize(np.array(y)[:, 1])
