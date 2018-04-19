# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:38:29 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
import mixture_distributions


if __name__ == '__main__':
    sampler = mixture_distributions.MixtureOfGaussians()
    z, x = sampler(10000, complete_data=True)
    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    K = len(np.unique(z))  # 3
    n = len(z)
    x_number = np.bincount(z)
    probabilities = [x_number/n]

    print("K:", K)
    print("p:", probabilities)

    x0_lst = []
    x1_lst = []
    x2_lst = []
    for i in np.arange(len(z)):
        if z[i] == 0:
            x0_lst.append(x[i])
        elif z[i] == 1:
            x1_lst.append(x[i])
        else:
            x2_lst.append(x[i])

    x0 = np.array(x0_lst)
    x1 = np.array(x1_lst)
    x2 = np.array(x2_lst)
    x_lst = []
    x_lst.append(x0)
    x_lst.append(x1)
    x_lst.append(x2)

    mean_lst = []
    var_lst = []
    std_lst = []
    for i in np.arange(K):
        n = len(x_lst[i])
        mean = np.array(x_lst[i]).sum() / n
        var = ((x_lst[i] - mean) ** 2).sum() / n
        std = var ** 0.5
        mean_lst.append(mean)
        var_lst.append(var)
        std_lst.append(std)

    y_lst = []
    for i in np.arange(K):
        y_lst.append(np.random.normal(mean_lst[i], std_lst[i], x_number[i]))

    data = []
    for i in np.arange(K):
        for _, y in enumerate(y_lst[i]):
            data.append(y)

    print("mean", mean_lst)
    print("var", var_lst)
    print("std", std_lst)
    sampler.visualize(np.array(data))
