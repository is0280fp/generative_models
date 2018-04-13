import numpy as np
import matplotlib.pyplot as plt
import basic_distributions


#def gf(sig_a, sig_b, mu_a, mu_b, x):
#    temp_a_lst = []
#    temp_b_lst = []
#
#    for i, x_i in enumerate(x):
#        c = sig_a * i + sig_b
#        mu_i = mu_a*i + mu_b
#
##        temp_a = i - i*np.exp(-c)*((x_i - mu_i)**2)
#        temp_a = i - np.exp(-c+np.log(i)+np.log((x_i - mu_i)**2))
#        temp_b = 1 - np.exp(-c)*(x_i - mu_i)**2
#        temp_a_lst.append(temp_a)
#        temp_b_lst.append(temp_b)
#
#    gy_a = -0.5 * np.sum(np.array(temp_a_lst))
#    gy_b = -0.5 * np.sum(np.array(temp_b_lst))
#    return gy_a, gy_b

def gf(sig_a, sig_b, mu_a, mu_b, x):
    i = np.arange(len(x))
    c = sig_a * i + sig_b
    mu = mu_a*i + mu_b

    temp_a = i - i*np.exp(-c)*((x - mu)**2)
    temp_b = 1 - np.exp(-c)*(x - mu)**2

    gy_a = -sig_a * np.sum(temp_a)
    gy_b = -sig_b * np.sum(temp_b)
    return gy_a, gy_b

#def f(sig_a, sig_b, mu_a, mu_b, x):
#    temp_lst = []
#
#    for i, x_i in enumerate(x):
#        c = sig_a * i + sig_b
#        mu_i = mu_a*i + mu_b
#
#        temp = np.log(2*np.pi) + c + np.exp(-c)*((x_i - mu_i)**2)
#        temp_lst.append(temp)
#    return -0.5 * np.sum(np.array(temp_lst))


def f(sig_a, sig_b, mu_a, mu_b, x):
    i = np.arange(len(x))
    c = sig_a * i + sig_b
    mu_i = mu_a*i + mu_b

    temp = np.log(2*np.pi) + c + np.exp(-0.5*c)*((x - mu_i)**2)
    return -0.5 * np.sum(temp)


if __name__ == '__main__':
    sampler = basic_distributions.LinearModel2()
    x = sampler()
    print("Distribution: ", sampler.get_name())
    print("Parameters: ", sampler.get_params())
    sampler.visualize(x)

    n = len(x)
    i = np.arange(n, dtype=np.int64)
    numerator = (i*x).sum() - (i.sum() * x.sum()) / n
    denominator = (i**2).sum() - (i.sum()**2) / n

    mean_a = numerator / denominator
    mean_b = (x.sum() - i.sum()*mean_a) / n

    #    初期値
    sig_a = 0.001
    sig_b = 0.01
    step_size = 0.0000000001
    sig_a_list = [sig_a]
    sig_b_list = [sig_b]
    gy_norm_list = []
    Y_list = []
    sig_a_size = 5
    sig_b_size = 10
    y_list = []

    _sig_a = np.linspace(0.00025, 0.00065, sig_a_size)
    _sig_b = np.linspace(0.005, 0.02, sig_b_size)
    grid_sig_a, grid_sig_b = np.meshgrid(_sig_a, _sig_b)
    for xxsig_a, xxsig_b in zip(grid_sig_a, grid_sig_b):
        for xsig_a, xsig_b in zip(xxsig_a, xxsig_b):
            Y_list.append(f(xsig_a, xsig_b, mean_a, mean_b, x))
    Y = np.reshape(Y_list, (sig_b_size, sig_a_size))

    for i in range(1, 1000):
        gy1, gy2 = gf(sig_a, sig_b, mean_a, mean_b, x)
        sig_a += step_size * gy1
        sig_b += step_size * gy2
        y = f(sig_a, sig_b, mean_a, mean_b, x)
        y_list.append(y)

        gy_norm = np.sqrt(gy1**2 + gy2**2)
        gy_norm_list.append(gy_norm)
        sig_a_list.append(sig_a)
        sig_b_list.append(sig_b)

#        plt.plot(sig_a_list, sig_b_list, ".-")
#        plt.contour(grid_sig_a, grid_sig_b, Y)
#        plt.xlim(_sig_a[0], _sig_a[-1])
#        plt.grid()
#        plt.colorbar()
##        plt.gca().set_aspect('equal')
#        plt.show()
#
#        plt.plot(y_list)
#        plt.grid()
#        plt.show()
#
##        plt.ylim(2700000000, 4000000000)
#        plt.plot(gy_norm_list)
#        plt.grid()
#        plt.show()

#        print("step_size", step_size)
#        print("gy_norm:", gy_norm)
#        print(i, sig_a, sig_b)
#        print("y:", y)

#    stds = sig_a*i + sig_b
    i = np.arange(len(x))
    means = mean_a * i + mean_b
    stds = np.exp(sig_a*i + sig_b)**0.5
    data = np.random.normal(means, stds, n)
    sampler.visualize(data)
