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
    i = np.arange(len(x))/1000
    lam = np.exp(mu_a*i + mu_b)
#
#    gy_a = np.sum((x/lam - 1) * lam*i)
#    gy_b = np.sum((x/lam - 1) * lam)

    gy_a = np.sum(x*i - lam*i)
    gy_b = np.sum(x - lam)
    return gy_a, gy_b


def f(mu_a, mu_b, x):
    i = np.arange(len(x))/1000
    temp = mu_a*i + mu_b
#    print("temp", temp)
    return np.sum(x*temp - np.exp(temp) - scipy.special.loggamma(x + 1).real)


if __name__ == '__main__':
    sampler = basic_distributions.PoissonGLM()
    x = sampler()
#    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())

    n = len(x)
    i = np.arange(n, dtype=np.int64)

    #    初期値
    mean_a = 0.001 + 0.01
    mean_b = 1 + 0.01/666
    step_size = 0.0001
    mean_a_list = [mean_a]
    mean_b_list = [mean_b]
    gy_norm_list = []
    mean_a_size = 30
    mean_b_size = 30
    y_list = []
    Y_list = []

    _mean_a = np.linspace(0.1, 3, mean_a_size)
    _mean_b = np.linspace(0.5, 1.7, mean_b_size)
    grid_mean_a, grid_mean_b = np.meshgrid(_mean_a, _mean_b)
    for xxmean_a, xxmean_b in zip(grid_mean_a, grid_mean_b):
        for xmean_a, xmean_b in zip(xxmean_a, xxmean_b):
            Y_list.append(f(xmean_a, xmean_b, x))
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

        plt.title("contour changing")
        plt.xlabel("mean_a")
        plt.ylabel("mean_b")
        plt.plot(mean_a_list, mean_b_list, ".-")
        plt.contour(grid_mean_a, grid_mean_b, Y)
        plt.grid()
        plt.colorbar()
        plt.show()

        plt.title("maximum  likelihood changing")
        plt.xlabel("step number")
        plt.ylabel("likelihood")
        plt.plot(y_list)
        plt.grid()
        plt.show()

        plt.title("gradient changing")
        plt.xlabel("step number")
        plt.ylabel("gradient norm")
        plt.plot(gy_norm_list)
        plt.grid()
        plt.show()

        print("step_size", step_size)
        print(i, mean_a, mean_b)
        print("gy1, gy2", gy1, gy2)
        print("gy_norm:", gy_norm)

    mean_a = mean_a / 1000
    print("mean_a, mean_b", mean_a, mean_b)
    i = np.arange(len(x))
    means = np.exp(mean_a * i + mean_b)
    y = np.random.poisson(means)
    sampler.visualize(y)
