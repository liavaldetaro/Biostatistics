import numpy as np
import scipy as sc
import scipy.stats as stats


def mean_CI_data(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sc.stats.sem(a)
    h = se * sc.stats.t.ppf((1 + confidence) / 2., n - 1)
    #h = se * 1.96
    return m, m - h, m + h


def mean_CI_model(mu, std, n, confidence=0.95):
    m = mu
    h = stats.norm.pdf((1 - confidence) / 2) * std / np.sqrt(n)
    return m, m - h, m + h


def mean_PI_data(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, std = np.mean(a), np.std(a)
    h = std * sc.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def mean_PI_model(mu, std, n, confidence=0.95):
    m = mu
    h = stats.norm.pdf((1 - confidence) / 2) * std
    return m, m - h, m + h

