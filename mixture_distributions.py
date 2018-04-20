# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:40:04 2015

@author: ryuhei
"""

import numpy as np
from basic_distributions import (
    ProbabilityDistribution,  Multinomial, Gaussian, Poisson
)


class MixtureDistributionBase(ProbabilityDistribution):
    def __init__(self, K, weights):
        assert K == len(weights)

        self.K = K
        self.weights = weights
        self.multinomial = Multinomial(weights)
        self.mixture_components = None

    def __call__(self, num_examples=10000, complete_data=False):
        z = []
        x = []
        for i in range(num_examples):
            z_i = self.multinomial(1)[0]
            x_i = self.mixture_components[z_i](1)[0]
            z.append(z_i)
            x.append(x_i)

        if complete_data:
            return (np.array(z), np.array(x))
        else:
            return np.array(x)


class MixtureOfGaussians(MixtureDistributionBase):
    def __init__(self, K=3, weights=[0.1, 0.7, 0.2],
                 means=[-15, 0, 30], stds=[1, 10, 2]):
        assert K == len(means) == len(stds)
        super(MixtureOfGaussians, self).__init__(K, weights)

        self.means = means
        self.stds = stds
        self.mixture_components = [
            Gaussian(means[k], stds[k]) for k in range(K)]


class MixtureOfPoissons(MixtureDistributionBase):
    def __init__(self, K=3, weights=[0.3, 0.3, 0.4],
                 means=[2, 20, 50]):
        assert K == len(means)
        super(MixtureOfPoissons, self).__init__(K, weights)

        self.means = means
        self.mixture_components = [Poisson(means[k]) for k in range(K)]


if __name__ == '__main__':
    distributions = [MixtureOfGaussians(),  # 0
                     MixtureOfPoissons(),   # 1
                     ]
    dist_type = 0
    complete_data = True

    sampler = distributions[dist_type]

    data = sampler(10000, complete_data)
    if complete_data:
        z, x = data
    else:
        x = data

    sampler.visualize(x)
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())