# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:38:29 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
import basic_distributions


if __name__ == '__main__':
    sampler = basic_distributions.Poisson()
    x = sampler()
    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    n = len(x)
    mean = x.sum() / n
    print("mean", mean)
    y = np.random.poisson(mean, n)
    sampler.visualize(y)
