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
    probabilities = np.bincount(z) / N

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
    means = np.array(mean_lst)
    stds = np.array(std_lst)

#  以下、モデル生成処理
    z_new = np.random.choice(K, N, p=probabilities)
    z_counts = np.bincount(z_new)
    x_new = np.random.normal(means[z_new], stds[z_new], N)

    print("probabilities", probabilities)
    print("mean", mean_lst)
    print("var", var_lst)
    print("std", std_lst)
    sampler.visualize(x_new)
