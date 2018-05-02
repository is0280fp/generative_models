# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:55:33 2015

@author: sakurai
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def generate_clustering_data():
    N, D, K = 100, 2, 4
    X = []

    X_k = np.random.randn(N, D) * [50, 1] + [0, 10] + [0, -100]
    X.append(X_k)

    X_k = np.random.randn(N, D) * [50, 1] + [0, -10] + [0, -100]
    X.append(X_k)

    X_k = np.random.randn(N, D) * [1, 50] + [10, 0] + [0, 100]
    X.append(X_k)

    X_k = np.random.randn(N, D) * [1, 50] + [-10, 0] + [0, 100]
    X.append(X_k)

    X = np.concatenate(X)
    np.random.shuffle(X)
    return X


if __name__ == '__main__':
    X = generate_clustering_data()

    plt.plot(X[:, 0], X[:, 1], '.')
    plt.title('Number of clusters is 4.')
    plt.show()
