# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:38:29 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
import basic_distributions


if __name__ == '__main__':
    sampler = basic_distributions.Multinomial()
    x = sampler()
    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    K = len(np.unique(x))  # 4
    n = len(x)
    p1 = list(x).count(0) / n
    p2 = list(x).count(1) / n
    p3 = list(x).count(2) / n
    p4 = list(x).count(3) / n
    probabilities = [p1, p2, p3, p4]

    print("K:", K)
    print("p:", probabilities)

    y = np.random.choice(K, n, p=probabilities)
    sampler.visualize(y)
