import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import pypoman
from scipy.stats import bernoulli, multinomial
from scipy.stats.mstats import gmean
import pytest
import coverage

from utils import Bets, Weights, mart, lower_confidence_bound, wright_lower_bound, \
    intersection_mart, plot_marts_eta, union_intersection_mart


def test_mart():
    sample = np.ones(10) * 0.5
    assert mart(sample, eta = 0.5, lam_func = Bets.lam_fixed, log = True) == np.log(1)
    assert mart(sample, eta = 0.5, lam_func = Bets.lam_fixed, log = False) == 1
    assert mart(sample, eta = 0.6, lam_func = Bets.lam_fixed, log = False) < 1
    assert mart(sample, eta = 0.4, lam_func = Bets.lam_fixed, log = False) > 1
    assert mart(sample, eta = 0.5, lam_func = Bets.lam_smooth, log = False) == 1
    assert mart(sample, eta = 0.5, lam_func = Bets.lam_smooth_predictable, log = False) == 1
    assert mart(sample, eta = 0.5, lam_func = Bets.lam_agrapa, log = False) == 1
    assert mart(sample, eta = 0.4, lam_func = Bets.lam_agrapa, log = False) > 1

def test_intersection_mart():
    #null is true
    sample = [np.ones(10) * 0.5, np.ones(10) * 0.5, np.ones(10) * 0.5]
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, combine = "product") == 0
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, theta_func = Weights.theta_fixed, combine = "sum") == 0
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, theta_func = Weights.theta_fixed, combine = "fisher") == 0
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, combine = "product", log = False) == 1
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, theta_func = Weights.theta_fixed, combine = "sum", log = False) == 1
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, combine = "fisher", log = False) == 1
    #alternative is true
    sample = [np.ones(10) * 0.6, np.ones(10) * 0.6, np.ones(10) * 0.6]
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, combine = "product") > 0
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, theta_func = Weights.theta_fixed, combine = "sum") > 0
    assert intersection_mart(sample, eta = [0.5, 0.5, 0.5], lam_func = Bets.lam_fixed, combine = "fisher") < 0 #note: this one is a P-value

def test_union_intersection_mart():
    N = [15, 15, 15]
    sample = [np.ones(5)*0.5, np.ones(5)*0.5, np.ones(5)*0.5]
    calX = [[0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]]
    assert union_intersection_mart(sample, N = N, eta_0 = 0.5, lam_func = Bets.lam_fixed, combine = "product", calX = calX)[0] <= 0
    assert union_intersection_mart(sample, N = N, eta_0 = 0.5, lam_func = Bets.lam_fixed, combine = "sum", theta_func = Weights.theta_fixed, calX = calX)[0] <= 0
    assert union_intersection_mart(sample, N = N, eta_0 = 0.5, lam_func = Bets.lam_fixed, combine = "fisher", calX = calX)[0] == 0
    assert union_intersection_mart(sample, N = N, eta_0 = 0.1, lam_func = Bets.lam_smooth, combine = "product", calX = calX)[0] >= 0

def test_lower_confidence_bound():
    sample_5 = np.ones(5) * 0.5
    sample_10 = np.ones(10) * 0.5
    assert lower_confidence_bound(sample_5, lam_func = Bets.lam_fixed, alpha = 0.01) < 0.5
    assert lower_confidence_bound(sample_5, lam_func = Bets.lam_fixed, alpha = 0.05) < 0.5
    assert lower_confidence_bound(sample_5, lam_func = Bets.lam_fixed, alpha = 0.70) < 0.5
    assert lower_confidence_bound(sample_5, lam_func = Bets.lam_agrapa, alpha = 0.05) < 0.5
    assert lower_confidence_bound(sample_10, lam_func = Bets.lam_fixed, alpha = 0.05) >= lower_confidence_bound(sample_5, lam_func = Bets.lam_fixed, alpha = 0.05)
    assert lower_confidence_bound(sample_5, lam_func = Bets.lam_fixed, alpha = 0.1) >= lower_confidence_bound(sample_5, lam_func = Bets.lam_fixed, alpha = 0.01)
    assert lower_confidence_bound(sample_5, lam_func = Bets.lam_agrapa, alpha = 0.1) >= lower_confidence_bound(sample_5, lam_func = Bets.lam_agrapa, alpha = 0.01)


def test_wright_lower_bound():
    N = [1000, 1000, 1000]
    samples = [0.5 * np.ones(50), 0.5 * np.ones(50), 0.5 * np.ones(50)]
    assert wright_lower_bound(x = samples, N = N, lam_func = Bets.lam_fixed, alpha = 0.05) < 0.5
    assert wright_lower_bound(x = samples, N = N, lam_func = Bets.lam_fixed, alpha = 0.05) > 0.2

    N = [5, 5, 3000]
    samples = [0.5 * np.ones(5), 0.5 * np.ones(5), 0.6 * np.ones(100)]
    assert wright_lower_bound(x = samples, N = N, lam_func = Bets.lam_fixed, alpha = 0.05) < 0.6
    assert wright_lower_bound(x = samples, N = N, lam_func = Bets.lam_fixed, alpha = 0.05) > 0.5
