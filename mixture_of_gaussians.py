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
            x0_lst.append([i, x[i]])
        elif z[i] == 1:
            x1_lst.append([i, x[i]])
        else:
            x2_lst.append([i, x[i]])

    mean_lst = []
    var_lst = []
    std_lst = []
    x0_lst = np.array(x0_lst)
    x1_lst = np.array(x1_lst)
    x2_lst = np.array(x2_lst)

    n = len(x0_lst[:, 1])
    mean = x0_lst[:, 1].sum() / n
    var = ((x0_lst[:, 1] - mean) ** 2).sum() / n
    std = var ** 0.5
    mean_lst.append(mean)
    var_lst.append(var)
    std_lst.append(std)

    n = len(x1_lst[:, 1])
    mean = x1_lst[:, 1].sum() / n
    var = ((x1_lst[:, 1] - mean) ** 2).sum() / n
    std = var ** 0.5
    mean_lst.append(mean)
    var_lst.append(var)
    std_lst.append(std)

    n = len(x2_lst[:, 1])
    mean = x2_lst[:, 1].sum() / n
    var = ((x2_lst[:, 1] - mean) ** 2).sum() / n
    std = var ** 0.5
    mean_lst.append(mean)
    var_lst.append(var)
    std_lst.append(std)

    y0_lst = []
    y1_lst = []
    y2_lst = []
    for i in np.arange(len(z)):
        if z[i] == 0:
            y0_lst.append([i, np.random.normal(mean_lst[0], std_lst[0], 1)])
        elif z[i] == 1:
            y1_lst.append([i, np.random.normal(mean_lst[1], std_lst[1], 1)])
        else:
            y2_lst.append([i, np.random.normal(mean_lst[2], std_lst[2], 1)])

#    print("mean", mean_lst)
#    print("var", var_lst)
#    print("std", std_lst)
#    sampler.visualize(np.array(data))
