# -*- coding: utf-8 -*-

# required
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def histogram():
    mu = 100
    sigma = 15
    x = mu + sigma * np.random.randn(10000)

    num_bins = 2
    n, bins, patches = plt.hist(x, num_bins, normed=1, color='blue', alpha=0.5)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'g')
    plt.subplots_adjust(left=0.1)
    plt.show()

def main():
    histogram()

if __name__ == "__main__":
    main()
