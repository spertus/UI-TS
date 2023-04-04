import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import pypoman
from scipy.stats import bernoulli, multinomial
from scipy.stats.mstats import gmean
import pytest
import coverage

from utils import Bets, Weights, Allocations, mart, selector, lower_confidence_bound, wright_lower_bound, \
    intersection_mart, plot_marts_eta, union_intersection_mart, construct_eta_grid,\
    construct_eta_grid_plurcomp


def test_mart():
    sample = np.ones(10) * 0.5
    assert len(mart(sample, eta = 0.5, lam_func = Bets.fixed, log = True)) == 11
    assert mart(sample, eta = 0.5, lam_func = Bets.fixed, log = True)[-1] == np.log(1)
    assert mart(sample, eta = 0.5, lam_func = Bets.fixed, log = False)[-1] == 1
    assert mart(sample, eta = 0.6, lam_func = Bets.fixed, log = False)[-1] < 1
    assert mart(sample, eta = 0.4, lam_func = Bets.fixed, log = False)[-1] > 1
    assert mart(sample, eta = 0.5, lam_func = Bets.smooth, log = False)[-1] == 1
    assert mart(sample, eta = 0.5, lam_func = Bets.smooth_predictable, log = False)[-1] == 1
    assert mart(sample, eta = 0.5, lam_func = Bets.agrapa, log = False)[-1] == 1
    assert mart(sample, eta = 0.4, lam_func = Bets.agrapa, log = False)[-1] > 1


def test_lower_confidence_bound():
    sample_5 = np.ones(5) * 0.5
    sample_10 = np.ones(10) * 0.5
    assert len(lower_confidence_bound(sample_5, lam_func = Bets.fixed, alpha = 0.01)) == 6
    assert len(lower_confidence_bound(sample_10, lam_func = Bets.fixed, alpha = 0.01)) == 11
    assert lower_confidence_bound(sample_5, lam_func = Bets.fixed, alpha = 0.01)[-1] < 0.5
    assert lower_confidence_bound(sample_5, lam_func = Bets.fixed, alpha = 0.05)[-1] < 0.5
    assert lower_confidence_bound(sample_5, lam_func = Bets.fixed, alpha = 0.70)[-1] < 0.5
    assert lower_confidence_bound(sample_5, lam_func = Bets.agrapa, alpha = 0.05)[-1] < 0.5
    assert lower_confidence_bound(sample_10, lam_func = Bets.fixed, alpha = 0.05)[-1] >= lower_confidence_bound(sample_5, lam_func = Bets.fixed, alpha = 0.05)[-1]
    assert lower_confidence_bound(sample_5, lam_func = Bets.fixed, alpha = 0.1)[-1] >= lower_confidence_bound(sample_5, lam_func = Bets.fixed, alpha = 0.01)[-1]
    assert lower_confidence_bound(sample_5, lam_func = Bets.agrapa, alpha = 0.1)[-1] >= lower_confidence_bound(sample_5, lam_func = Bets.agrapa, alpha = 0.01)[-1]


def test_wright_lower_bound():
    N = [1000, 1000, 1000]
    samples = [0.5 * np.ones(50), 0.5 * np.ones(50), 0.5 * np.ones(50)]
    assert wright_lower_bound(samples, N, Bets.fixed, Allocations.round_robin, 0.05)[-1] < 0.5
    assert wright_lower_bound(samples, N, Bets.fixed, Allocations.round_robin, 0.05)[-1] > 0.2

    N = [5, 5, 3000]
    samples = [0.5 * np.ones(5), 0.5 * np.ones(5), 0.6 * np.ones(100)]
    assert wright_lower_bound(samples, N, Bets.fixed, Allocations.proportional_round_robin, 0.05)[-1] < 0.6
    assert wright_lower_bound(samples, N, Bets.fixed, Allocations.proportional_round_robin, 0.05)[-1] > 0.5

def test_selector():
    N = [1000, 1000, 1000]
    n = [50, 50, 50]
    samples = [0.5 * np.ones(n[0]), 0.5 * np.ones(n[1]), 0.5 * np.ones(n[2])]
    assert selector(samples, N, Allocations.round_robin).shape[0] == 151
    assert selector(samples, N, Allocations.round_robin).shape[1] == 3
    np.testing.assert_array_equal(selector(samples, N, Allocations.round_robin)[-1,:], [50,50,50])

    n = [100, 100, 50]
    samples = [0.5 * np.ones(n[0]), 0.5 * np.ones(n[1]), 0.5 * np.ones(n[2])]
    assert selector(samples, N, Allocations.round_robin).shape[0] == 251
    assert selector(samples, N, Allocations.round_robin).shape[1] == 3
    np.testing.assert_array_equal(selector(samples, N, Allocations.proportional_round_robin)[-1,:], [100,100,50])

    N = [1000, 2000, 3000]
    n = [1000, 2000, 3000]
    samples = [0.5 * np.ones(n[0]), 0.5 * np.ones(n[1]), 0.5 * np.ones(n[2])]
    assert selector(samples, N, Allocations.proportional_round_robin).shape[0] == 6001
    assert selector(samples, N, Allocations.proportional_round_robin).shape[1] == 3
    np.testing.assert_array_equal(selector(samples, N, Allocations.proportional_round_robin)[-1,:], [1000, 2000, 3000])
    np.testing.assert_array_equal(selector(samples, N, Allocations.proportional_round_robin)[3000,:], [500, 1000, 1500])


def test_intersection_mart():
    #null is true
    N = [10,10,10]
    sample = [np.ones(N[0]) * 0.5, np.ones(N[1]) * 0.5, np.ones(N[2]) * 0.5]
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "fisher")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product", log = False)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum", log = False)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher", log = False)[-1] == 1
    #alternative is true
    sample = [np.ones(N[0]) * 0.6, np.ones(N[1]) * 0.6, np.ones(N[2]) * 0.6]
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product")[-1] > 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum")[-1] > 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher")[-1] < 0 #note: this one is a P-value

def test_construct_eta_grid():
    N = [15, 15, 15]
    calX = [[0, 0.5, 1], [0, 0.5, 1], [0, 0.5, 1]]
    etas = construct_eta_grid(eta_0 = 0.5, N = N, calX = calX)[0]
    assert etas.count((0.5, 0.5, 0.5)) == 1
    assert etas.count((0, 0.5, 1)) == 1
    assert etas.count((2/3, 1/3, 0)) == 1
    assert etas.count((1, 1, 1)) == 0

def test_construct_eta_grid_plurcomp():
    N = [15, 15]
    etas = construct_eta_grid_plurcomp(N = N, A_c = [1, 0.5])[0]
    assert etas.count((0, 1.5)) == 1
    assert etas.count((0.5, 1)) == 1
    assert etas.count((0.25, 1.25)) == 0

def test_union_intersection_mart():
    N = [5, 5, 5]
    sample = [np.ones(N[0])*0.5, np.ones(N[1])*0.5, np.ones(N[2])*0.5]
    etas = [(0, 0.5, 1), (0.5, 0.5, 0.5)]
    assert all(union_intersection_mart(sample, N, etas, Bets.fixed, Allocations.round_robin, combine = "product")[0] <= 0)
    assert all(union_intersection_mart(sample, N, etas, Bets.fixed, Allocations.round_robin, combine = "sum", theta_func = Weights.fixed)[0] <= 0)
    assert all(union_intersection_mart(sample, N, etas, Bets.fixed, Allocations.round_robin, combine = "fisher")[0] == 0)
    assert all(union_intersection_mart(sample, N, etas, Bets.smooth, Allocations.round_robin, combine = "product")[0] >= 0)
