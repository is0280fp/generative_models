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

    means = np.array(mean_lst)
    stds = np.array(std_lst)
    y = np.random.normal(means[Z], stds[Z], N)

#    y = sorted(y, key=itemgetter(0))
    print("probabilities", probabilities)
    print("mean", mean_lst)
    print("var", var_lst)
    print("std", std_lst)
    sampler.visualize(y)
