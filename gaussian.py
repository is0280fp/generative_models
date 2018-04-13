# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:38:29 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
import basic_distributions


if __name__ == '__main__':
    sampler = basic_distributions.Gaussian()
    x = sampler()
    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    n = len(x)
    mean = x.sum() / n
    var = ((x - mean) ** 2).sum() / n
    std = var ** 0.5
    print("mean", mean)
    print("var", var)
    print("std", std)
    y = np.random.normal(mean, std, n)
    sampler.visualize(y)
