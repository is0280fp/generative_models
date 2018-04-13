# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:38:29 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
import basic_distributions


if __name__ == '__main__':
    sampler = basic_distributions.LinearModel()
    x = sampler()
    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    n = len(x)
    i = np.arange(n, dtype=np.int64)
    numerator = (i*x).sum() - (i.sum() * x.sum()) / n
    denominator =  (i**2).sum() - (i.sum()**2) / n

    mean_a = numerator / denominator

    mean_b = (x.sum() - i.sum()*mean_a) / n

    c = (x**2).sum() - 2*mean_a*((i*x).sum()) - 2*mean_b*(
            x.sum()) + (mean_a**2)*((i**2).sum()) + 2*mean_a*mean_b*(
                    i.sum()) + n*(mean_b**2)

    var = n / c
    std = var**0.5

    means = mean_a * i + mean_b
    print("mean_a", mean_a)
    print("mean_b", mean_b)
#    print("means", means)
    print("std", std)
    y = np.random.normal(means, std, n)
    sampler.visualize(y)
