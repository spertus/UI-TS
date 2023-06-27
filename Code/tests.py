import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
import pypoman
from scipy.stats import bernoulli, multinomial
from scipy.stats.mstats import gmean
import pytest
import coverage

from utils import Bets, Weights, Allocations, mart, selector, lower_confidence_bound, global_lower_bound, \
    intersection_mart, plot_marts_eta, union_intersection_mart, construct_eta_grid,\
    construct_eta_grid_plurcomp, construct_vertex_etas, simulate_comparison_audit,\
    random_truncated_gaussian, PGD, negexp_ui_mart


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
    #WOR
    assert mart(sample, eta = 0.5, N = 15, lam_func = Bets.fixed, log = False)[-1] == 1
    assert mart(sample, eta = 0.4, N = 15, lam_func = Bets.agrapa, log = False)[-1] > 1
    assert mart(sample, eta = 0.1, N = 15, lam_func = Bets.fixed, log = False)[-1] == np.inf



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
    assert lower_confidence_bound(sample_5, lam_func = Bets.agrapa, alpha = 0.05, N = 10)[-1] <= 0.5
    assert lower_confidence_bound(sample_5, lam_func = Bets.agrapa, alpha = 0.05, N = 5)[-1] >= 0.4


def test_global_lower_bound():
    N = [1000, 1000, 1000]
    samples = [0.5 * np.ones(50), 0.5 * np.ones(50), 0.5 * np.ones(50)]
    assert global_lower_bound(samples, N, Bets.fixed, Allocations.round_robin, 0.05)[-1] < 0.5
    assert global_lower_bound(samples, N, Bets.fixed, Allocations.round_robin, 0.05)[-1] > 0.2
    #without replacement
    assert global_lower_bound(samples, N, Bets.fixed, Allocations.round_robin, 0.05, WOR = True)[-1] < 0.5
    assert global_lower_bound(samples, N, Bets.fixed, Allocations.round_robin, 0.05, WOR = True)[-1] > 0.2
    assert global_lower_bound(samples, N, Bets.fixed, Allocations.round_robin, 0.05, WOR = True)[-1] > global_lower_bound(samples, N, Bets.fixed, Allocations.round_robin, 0.05, WOR = False)[-1]

    N = [5, 5, 3000]
    samples = [0.5 * np.ones(5), 0.5 * np.ones(5), 0.6 * np.ones(100)]
    assert global_lower_bound(samples, N, Bets.fixed, Allocations.proportional_round_robin, 0.05)[-1] < 0.6
    assert global_lower_bound(samples, N, Bets.fixed, Allocations.proportional_round_robin, 0.05)[-1] > 0.5
    assert global_lower_bound(samples, N, Bets.fixed, Allocations.proportional_round_robin, 0.05, WOR = True)[-1] > global_lower_bound(samples, N, Bets.fixed, Allocations.proportional_round_robin, 0.05, WOR = False)[-1]


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

    N = [1000, 2000, 3000]
    n = [1000, 2000, 3000]
    eta = [0.5, 0.5, 0.5]
    samples = [0.8 * np.ones(n[0]), 0.5 * np.ones(n[1]), 0.2 * np.ones(n[2])]
    assert selector(samples, N, Allocations.proportional_to_mart, eta, Bets.fixed).shape[0] == 6001
    assert selector(samples, N, Allocations.proportional_to_mart, eta, Bets.fixed).shape[1] == 3
    np.testing.assert_array_equal(selector(samples, N, Allocations.proportional_to_mart, eta, Bets.fixed)[-1,:], [1000, 2000, 3000])
    #check whether the first stratum is preferentially sampled
    selections = selector(samples, N, Allocations.proportional_to_mart, eta, Bets.fixed)
    assert selections[1000,0] > selections[1000,2]


def test_intersection_mart():
    #null is true
    N = [10,10,5]
    sample = [np.ones(N[0]) * 0.5, np.ones(N[1]) * 0.5, np.ones(N[2]) * 0.5]
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "fisher")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product", log = False)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum", log = False)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher", log = False)[-1] == 1
    #without replacement
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product", WOR = True)[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum", WOR = True)[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "fisher", WOR = True)[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product", log = False, WOR = True)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum", log = False, WOR = True)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher", log = False, WOR = True)[-1] == 1
    #different allocation functions
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.neyman, combine = "product", log = False, WOR = True)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.more_to_larger_means, combine = "product", log = False, WOR = True)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.proportional_to_mart, combine = "fisher", log = False, WOR = True)[-1] == 1
    #mixing distribution
    md = np.array([[0.5,0.5,0.5], [0.25, 0.5, 0.75]])
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], mixing_dist = md, allocation_func = Allocations.round_robin, combine = "product", WOR = False)[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], mixing_dist = md, allocation_func = Allocations.round_robin, combine = "product", log=False, WOR = True)[-1] == 1

    #alternative is true
    sample = [np.ones(N[0]) * 0.6, np.ones(N[1]) * 0.6, np.ones(N[2]) * 0.6]
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product")[-1] > 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum")[-1] > 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher")[-1] < 0
    #alternative it true, without replacement
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product", WOR = True)[-1] > 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum", WOR = True)[-1] > 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher", WOR = True)[-1] < 0
    md = np.array([[0.5,0.5,0.5], [0.25, 0.5, 0.75], [1,1,1]])
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], mixing_dist = md, allocation_func = Allocations.round_robin, combine = "product", WOR = False)[-1] > 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], mixing_dist = md, allocation_func = Allocations.round_robin, combine = "product", log=False, WOR = True)[-1] > 1


def test_construct_eta_grid():
    N = [15, 15, 15]
    calX = [np.array([0, 0.5, 1]), np.array([0, 0.5, 1]), np.array([0, 0.5, 1])]
    etas = construct_eta_grid(eta_0 = 0.5, N = N, calX = calX)[0]
    assert etas.count((0.5, 0.5, 0.5)) == 1
    assert etas.count((0, 0.5, 1)) == 1
    assert etas.count((1, 1, 1)) == 0

def test_construct_eta_grid_plurcomp():
    N = [15, 15]
    etas = construct_eta_grid_plurcomp(N = N, A_c = [1, 0.5])[0]
    assert etas.count((0, 0.75)) == 1
    assert etas.count((0.25, 0.5)) == 1
    assert etas.count((0.125, 0.625)) == 0

def test_construct_vertex_etas():
    assert construct_vertex_etas(N = [10000, 10000], eta_0 = 1/2).count((1,0)) == 1
    assert construct_vertex_etas(N = [10000, 10000], eta_0 = 1/2).count((0,1)) == 1
    assert len(construct_vertex_etas(N = [10, 10], eta_0 = 1/2)) == 2
    assert len(construct_vertex_etas(N = [10, 10, 10], eta_0 = 1/2)) == 6
    assert len(construct_vertex_etas(N = [10, 10, 10, 10], eta_0 = 1/2)) == 6
    assert len(construct_vertex_etas(N = [10, 10, 10, 10, 10], eta_0 = 1/2)) == 30
    assert len(construct_vertex_etas(N = [10, 10, 10, 10, 10, 10], eta_0 = 1/2)) == 20


def test_union_intersection_mart():
    N = [5, 5, 5]
    sample = [np.ones(N[0])*0.5, np.ones(N[1])*0.5, np.ones(N[2])*0.5]
    etas = [(0, 0.5, 1), (0.5, 0.5, 0.5)]
    assert all(union_intersection_mart(sample, N, etas, Bets.fixed, allocation_func = Allocations.round_robin, combine = "product")[0] <= 0)
    assert all(union_intersection_mart(sample, N, etas, Bets.fixed, allocation_func = Allocations.round_robin, combine = "sum", theta_func = Weights.fixed)[0] <= 0)
    assert all(union_intersection_mart(sample, N, etas, Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher")[0] == 0)
    assert all(union_intersection_mart(sample, N, etas, Bets.smooth, allocation_func = Allocations.round_robin, combine = "product")[0] >= 0)
    # check mixture distributions
    assert all(union_intersection_mart(sample, N, etas, allocation_func = Allocations.round_robin, mixture = "vertex", combine = "product")[0] <= 0)
    assert all(union_intersection_mart(sample, N, etas, allocation_func = Allocations.round_robin, mixture = "uniform", combine = "product")[0] <= 0)

def test_simulate_comparison_audit():
    N = [20, 20]
    A_c = [0.8, 0.8]
    p_1 = [0.0, 0.0]
    p_2 = [0.0, 0.0]
    assert 1 < simulate_comparison_audit(N, A_c, p_1, p_2, lam_func = Bets.fixed, allocation_func = Allocations.round_robin, WOR = True, reps = 1) < 40
    assert 1 < simulate_comparison_audit(N, A_c, p_1, p_2, lam_func = Bets.fixed, allocation_func = Allocations.proportional_to_mart, WOR = True, reps = 10)
    assert 1 < simulate_comparison_audit(N, A_c, p_1, p_2, lam_func = None, allocation_func = Allocations.proportional_to_mart, mixture = "uniform", WOR = True, reps = 10)


def test_random_truncated_gaussian():
    assert len(random_truncated_gaussian(0.5, 0.1, 30)) == 30
    samples = random_truncated_gaussian(0.5, 1, 20)
    assert ((0 < samples) & (samples < 1)).all()
    assert 0.4 < random_truncated_gaussian(0.5, 0.001, 1) < 0.6


def test_negexp_ui_mart():
    #these tests are probabilistic, they may sometimes fail (but should rarely)
    N = [100, 50]
    x_null_1 = [random_truncated_gaussian(0.5, 0.05, N[0]), random_truncated_gaussian(0.5, 0.05, N[1])]
    assert np.max(negexp_ui_mart(x_null_1, N, Allocations.round_robin, eta_0 = 0.5)) < np.log(100) #there should be less than 1% chance this doesnt happen
    x_null_2 = [random_truncated_gaussian(0.2, 0.05, N[0]), random_truncated_gaussian(0.8, 0.05, N[1])]
    assert np.max(negexp_ui_mart(x_null_2, N, Allocations.round_robin, eta_0 = 0.5)) < np.log(100)
    assert np.max(negexp_ui_mart(x_null_2, N, Allocations.more_to_larger_means, eta_0 = 0.5)) < np.log(100)
    x_null_3 = [random_truncated_gaussian(0.4, 0.05, N[0]), random_truncated_gaussian(0.6, 0.05, N[1])]
    assert np.max(negexp_ui_mart(x_null_2, N, Allocations.round_robin, eta_0 = 0.5)) < np.log(100)


    #test that it does reject eventually under alternative
    x_alt_1 = [random_truncated_gaussian(0.8, 0.05, N[0]), random_truncated_gaussian(0.8, 0.05, N[1])]
    assert np.max(negexp_ui_mart(x_alt_1, N, Allocations.round_robin, eta_0 = 0.5)) > np.log(20)
    x_alt_2 = [random_truncated_gaussian(0.4, 0.05, N[0]), random_truncated_gaussian(0.8, 0.05, N[1])]
    assert np.max(negexp_ui_mart(x_alt_2, N, Allocations.more_to_larger_means, eta_0 = 0.5)) > np.log(20)

    #check PGD works for higher dimensions
    K = 5
    N = [100 for _ in range(K)]
    x_alt_1 = [random_truncated_gaussian(0.8, 0.05, N[k]) for k in range (K)]
    assert np.max(negexp_ui_mart(x_alt_1, N, Allocations.round_robin, eta_0 = 0.5)) > np.log(20)
