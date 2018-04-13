# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 19:38:29 2018

@author: yume
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import basic_distributions


def gf(mu_a, mu_b, x):
    i = np.arange(len(x))
    lam = np.exp(mu_a*i + mu_b)

    gy_a = np.sum((x/lam - 1) * lam*i)
    gy_b = np.sum((x/lam - 1) * lam)
    return gy_a, gy_b


def f(mu_a, mu_b, x):
    i = np.arange(len(x))
    c = mu_a*i + mu_b
    return np.sum(x*c - np.exp(c) - scipy.special.loggamma(x + 1).real)


if __name__ == '__main__':
    sampler = basic_distributions.PoissonGLM()
    x = sampler()
#    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    n = len(x)
    i = np.arange(n, dtype=np.int64)

    #    初期値
    mean_a = 0.003
    mean_b = 0.1
    step_size = 0.0000000001
    mean_a_list = [mean_a]
    mean_b_list = [mean_b]
    gy_norm_list = []
    mean_a_size = 5
    mean_b_size = 10
    y_list = []
    Y_list = []

    _mean_a = np.linspace(0.00025, 0.00065, mean_a_size)
    _mean_b = np.linspace(0.005, 1.5, mean_b_size)
    grid_mean_a, grid_mean_b = np.meshgrid(_mean_a, _mean_b)
    for xxmean_a, xxmean_b in zip(grid_mean_a, grid_mean_b):
        for xmean_a, xmean_b in zip(xxmean_a, xxmean_b):
            Y_list.append(f(mean_a, mean_b, x))
    Y = np.reshape(Y_list, (mean_b_size, mean_a_size))

    for i in range(1, 1000):
        gy1, gy2 = gf(mean_a, mean_b, x)
        mean_a += step_size * gy1
        mean_b += step_size * gy2
        y = f(mean_a, mean_b, x)
        y_list.append(y)

        gy_norm = np.sqrt(gy1**2 + gy2**2)
        gy_norm_list.append(gy_norm)
        mean_a_list.append(mean_a)
        mean_b_list.append(mean_b)

        plt.plot(mean_a_list, mean_b_list, ".-")
        plt.contour(grid_mean_a, grid_mean_b, Y)
        plt.xlim(_mean_a[0], _mean_a[-1])
        plt.grid()
        plt.colorbar()
#        plt.gca().set_aspect('equal')
        plt.show()

        plt.plot(y_list)
        plt.grid()
        plt.show()

        plt.plot(gy_norm_list)
        plt.grid()
        plt.show()

        print("step_size", step_size)
        print("gy_norm:", gy_norm)
        print(i, mean_a, mean_b)
        print("y:", y)

#    print("mean_a, mean_b", mean_a, mean_b)
#    i = np.arange(len(x))
#    means = np.exp(mean_a * i + mean_b)
#    y = np.random.poisson(means)
#    sampler.visualize(y)
