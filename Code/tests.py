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
    intersection_mart, plot_marts_eta, brute_force_uits, construct_exhaustive_eta_grid,\
    construct_eta_grid_plurcomp, construct_vertex_etas, simulate_plurcomp,\
    random_truncated_gaussian, PGD, convex_uits, construct_eta_bands, banded_uits,\
    generate_hybrid_audit_population, generate_oneaudit_population


def test_mart():
    sample = np.ones(10) * 0.5
    assert len(mart(sample, eta = 0.5, lam_func = Bets.fixed, log = True)) == 11
    assert mart(sample, eta = 0.5, lam_func = Bets.fixed, log = True)[-1] == np.log(1)
    assert mart(sample, eta = 0.5, lam_func = Bets.fixed, log = False)[-1] == 1
    assert mart(sample, eta = 0.6, lam_func = Bets.fixed, log = False)[-1] < 1
    assert mart(sample, eta = 0.4, lam_func = Bets.fixed, log = False)[-1] > 1
    assert mart(sample, eta = 0.5, lam_func = Bets.negative_exponential, log = False)[-1] == 1
    assert mart(sample, eta = 0.5, lam_func = Bets.agrapa, log = False)[-1] == 1
    assert mart(sample, eta = 0.4, lam_func = Bets.agrapa, log = False)[-1] > 1
    assert mart(sample, eta = 0.4, lam_func = Bets.predictable_plugin, log = False)[-1] > 1
    #test kwargs
    agrapa = lambda x, eta: Bets.agrapa(x, eta, c = 0.9, sd_min = 0.2) #is there an easier way to specify?
    assert mart(sample, eta = 0.5, lam_func = agrapa, log = False)[-1] == 1
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


def test_convex_bets():
    N = 10
    eta = 0.5
    x = 0.5 * np.ones(N)

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

    #eta-adaptive methods
    N = [1000, 1000, 1000]
    n = [1000, 1000, 1000]
    eta = [0.5, 0.5, 0.5]
    samples = [0.8 * np.ones(n[0]), 0.5 * np.ones(n[1]), 0.2 * np.ones(n[2])]
    bets = [Bets.fixed(samples[k], eta[k]) for k in np.arange(3)]
    assert selector(samples, N, Allocations.proportional_to_mart, eta, bets).shape[0] == 3001
    assert selector(samples, N, Allocations.proportional_to_mart, eta, bets).shape[1] == 3
    np.testing.assert_array_equal(selector(samples, N, Allocations.proportional_to_mart, eta, bets)[-1,:], [1000, 1000, 1000])
    #check whether the first stratum is preferentially sampled
    selections = selector(samples, N, Allocations.proportional_to_mart, eta, bets)
    assert selections[100,0] > selections[100,2]
    #same as above but for predictable_kelly
    assert selector(samples, N, Allocations.predictable_kelly, eta, bets).shape[0] == 3001
    assert selector(samples, N, Allocations.predictable_kelly, eta, bets).shape[1] == 3
    np.testing.assert_array_equal(selector(samples, N, Allocations.predictable_kelly, eta, bets)[-1,:], [1000, 1000, 1000])
    selections = selector(samples, N, Allocations.predictable_kelly, eta, bets)
    assert selections[100,0] > selections[100,2]
    #check whether predictable Kelly allocates more to strata where null is False
    N = [100, 100]
    n = [100, 100]
    eta_1 = [1,0]
    eta_2 = [0,1]
    samples = [0.6 * np.ones(n[0]), 0.6 * np.ones(n[1])]
    bets = [Bets.fixed(samples[k], eta[k]) for k in np.arange(2)]
    selections_1 = selector(samples, N, Allocations.predictable_kelly, eta_1, bets)
    selections_2 = selector(samples, N, Allocations.predictable_kelly, eta_2, bets)
    assert selections_1[10,1] > selections_2[10,1]
    assert selections_1[10,0] < selections_2[10,0]


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
    #when allocation is done outside the intersection martingale
    lam = [Bets.fixed(sample[k], 0.5) for k in np.arange(3)]
    T_k = selector(sample, N, allocation_func = Allocations.round_robin, eta = [0.5,0.5,0.5], lam = lam)
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, T_k = T_k, combine = "product", log = False, WOR = True, last = True) == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.negative_exponential, T_k = T_k, combine = "product", log = False, WOR = True, last = True) == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, T_k = T_k, combine = "product", log = False, WOR = False, last = False)[-1] == intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, T_k = T_k, combine = "product", log = False, WOR = False, last = True)
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, T_k = T_k, combine = "product", log = True, WOR = False, last = False)[-1] == intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, T_k = T_k, combine = "product", log = True, WOR = False, last = True)



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

    #test extreme points
    N = [20, 20]
    sample = [np.ones(N[0]) * .3, np.ones(N[0]) * .8]
    assert intersection_mart(sample, N, eta = [0, 1], lam_func = Bets.agrapa, allocation_func = Allocations.predictable_kelly, combine = "product", log = False, WOR = True)[-1] >= 1



def test_construct_eta_bands():
    N = [15, 15]
    eta_bands = construct_eta_bands(eta_0 = 0.5, N = N, n_bands = 100)
    etas = [list(eta_bands[i][0][0]) for i in np.arange(len(eta_bands))]
    assert etas.count([0.5, 0.5]) == 1
    assert etas.count([0, 1]) == 1
    assert etas.count([1, 1]) == 0

def test_construct_exhaustive_eta_grid():
    N = [15, 15, 15]
    calX = [np.array([0, 0.5, 1]), np.array([0, 0.5, 1]), np.array([0, 0.5, 1])]
    etas = construct_exhaustive_eta_grid(eta_0 = 0.5, N = N, calX = calX)[0]
    assert etas.count((0.5, 0.5, 0.5)) == 1
    assert etas.count((0, 0.5, 1)) == 1
    assert etas.count((1, 1, 1)) == 0

def test_construct_eta_grid_plurcomp():
    N = [15, 15]
    etas = construct_eta_grid_plurcomp(N = N, A_c = [1, 0.5], assorter_method = "sts")[0]
    assert etas.count((0, 0.75)) == 1
    assert etas.count((0.25, 0.5)) == 1
    assert etas.count((0.125, 0.625)) == 0
    etas = construct_eta_grid_plurcomp(N = N, A_c = [1, 0.5], assorter_method = "global")[0]
    assert etas.count((0.5,0.5)) == 1
    assert etas.count((0,1)) == 1
    assert etas.count((1,0)) == 1
    assert etas.count((1,1)) == 0

def test_construct_vertex_etas():
    assert construct_vertex_etas(N = [10000, 10000], eta_0 = 1/2).count((1,0)) == 1
    assert construct_vertex_etas(N = [10000, 10000], eta_0 = 1/2).count((0,1)) == 1
    assert len(construct_vertex_etas(N = [10, 10], eta_0 = 1/2)) == 2
    assert len(construct_vertex_etas(N = [10, 10, 10], eta_0 = 1/2)) == 6
    assert len(construct_vertex_etas(N = [10, 10, 10, 10], eta_0 = 1/2)) == 6
    assert len(construct_vertex_etas(N = [10, 10, 10, 10, 10], eta_0 = 1/2)) == 30
    assert len(construct_vertex_etas(N = [10, 10, 10, 10, 10, 10], eta_0 = 1/2)) == 20


def test_banded_uits():
    N = [15, 15]
    eta_bands_3 = construct_eta_bands(eta_0 = 0.5, N = N, n_bands = 3)
    eta_bands_100 = construct_eta_bands(eta_0 = 0.5, N = N, n_bands = 100)
    #null is true
    sample = [np.ones(N[0])*0.5, np.ones(N[1])*0.5]
    assert all(banded_uits(sample, N, eta_bands_3, Bets.agrapa, allocation_func = Allocations.round_robin)[0] <= 0)
    assert all(banded_uits(sample, N, eta_bands_3, Bets.fixed, allocation_func = Allocations.round_robin, WOR = False)[0] <= 0)
    assert all(banded_uits(sample, N, eta_bands_100, Bets.agrapa, allocation_func = Allocations.round_robin, WOR = True)[0] <= 0)
    assert all(banded_uits(sample, N, eta_bands_100, Bets.fixed, allocation_func = Allocations.predictable_kelly, WOR = False)[0] <= 0)
    assert all(banded_uits(sample, N, eta_bands_100, Bets.fixed, allocation_func = Allocations.greedy_kelly, WOR = True)[0] <= 0)
    #null is false
    sample = [np.ones(N[0])*0.5, np.ones(N[1])]
    assert banded_uits(sample, N, eta_bands_3, Bets.agrapa, allocation_func = Allocations.round_robin)[0][-1] >= 0
    assert banded_uits(sample, N, eta_bands_3, Bets.agrapa, allocation_func = Allocations.round_robin, WOR = True)[0][-1] >= 0
    assert banded_uits(sample, N, eta_bands_3, Bets.negative_exponential, allocation_func = Allocations.round_robin, WOR = True)[0][-1] >= 0
    assert banded_uits(sample, N, eta_bands_100, Bets.agrapa, allocation_func = Allocations.round_robin, WOR = False)[0][-1] >= 0
    assert banded_uits(sample, N, eta_bands_100, Bets.agrapa, allocation_func = Allocations.predictable_kelly, WOR = True)[0][-1] >= 0
    assert banded_uits(sample, N, eta_bands_100, Bets.agrapa, allocation_func = Allocations.greedy_kelly, WOR = False)[0][-1] >= 0
    assert banded_uits(sample, N, eta_bands_100, Bets.negative_exponential, allocation_func = Allocations.greedy_kelly, WOR = True)[0][-1] >= 0
    assert banded_uits(sample, N, eta_bands_100, Bets.inverse_eta, allocation_func = Allocations.greedy_kelly, WOR = True)[0][-1] >= 0


def test_brute_force_uits():
    N = [5, 5, 5]
    sample = [np.ones(N[0])*0.5, np.ones(N[1])*0.5, np.ones(N[2])*0.5]
    etas = [(0, 0.5, 1), (0.5, 0.5, 0.5)]
    assert all(brute_force_uits(sample, N, etas, Bets.fixed, allocation_func = Allocations.round_robin, combine = "product")[0] <= 0)
    assert all(brute_force_uits(sample, N, etas, Bets.fixed, allocation_func = Allocations.round_robin, combine = "sum", theta_func = Weights.fixed)[0] <= 0)
    assert all(brute_force_uits(sample, N, etas, Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher")[0] <= 0)
    assert all(brute_force_uits(sample, N, etas, Bets.agrapa, allocation_func = Allocations.greedy_kelly, combine = "product")[0] <= 0)
    # check mixture distributions
    assert all(brute_force_uits(sample, N, etas, allocation_func = Allocations.round_robin, mixture = "vertex", combine = "product")[0] <= 0)
    assert all(brute_force_uits(sample, N, etas, allocation_func = Allocations.round_robin, mixture = "uniform", combine = "product")[0] <= 0)


def test_simulate_plurcomp():
    N = [40, 40]
    A_c = [0.8, 0.8]
    p_1 = [0.0, 0.0]
    p_2 = [0.0, 0.0]

    #lcb
    #check global stopping times
    assert 1 < simulate_plurcomp(N, A_c, p_1, p_2, lam_func = Bets.agrapa, allocation_func = Allocations.round_robin, method = "lcb", WOR = False, reps = 2)[0] < 80
    #check global sample sizes
    assert 1 < simulate_plurcomp(N, A_c, p_1, p_2, lam_func = Bets.fixed, allocation_func = Allocations.round_robin, method = "lcb", WOR = False, reps = 1)[1] < 80

    #ui-ts
    #check global stopping times
    assert 1 < simulate_plurcomp(N, A_c, p_1, p_2, lam_func = Bets.agrapa, allocation_func = Allocations.round_robin, WOR = True, reps = 2)[0] < 80
    #check global sample sizes
    assert 1 < simulate_plurcomp(N, A_c, p_1, p_2, lam_func = Bets.fixed, allocation_func = Allocations.round_robin, WOR = True, reps = 1)[1] < 80
    #check if sample size is larger than stopping time
    g_st, g_ss = simulate_plurcomp(N, A_c, p_1, p_2, lam_func = Bets.fixed, allocation_func = Allocations.predictable_kelly, WOR = True, reps = 1)
    assert g_st < g_ss

    # different alternative
    N = [20, 20]
    A_c = [0.4, 0.8]
    p_1 = [0.0, 0.0]
    p_2 = [0.0, 0.0]
    #check global stopping times
    assert 1 < simulate_plurcomp(N, A_c, p_1, p_2, lam_func = Bets.inverse_eta, allocation_func = Allocations.round_robin, WOR = True, reps = 1)[0] < 40
    #check global sample sizes
    assert 1 < simulate_plurcomp(N, A_c, p_1, p_2, lam_func = Bets.agrapa, allocation_func = Allocations.round_robin, WOR = True, reps = 1)[1] < 40


def test_random_truncated_gaussian():
    assert len(random_truncated_gaussian(0.5, 0.1, 30)) == 30
    samples = random_truncated_gaussian(0.5, 1, 20)
    assert ((0 < samples) & (samples < 1)).all()
    assert 0.4 < random_truncated_gaussian(0.5, 0.001, 1) < 0.6


def test_generate_hybrid_audit_population():
    # basic STS hybrid audit
    pop = generate_hybrid_audit_population(N = [200, 200], A_c = [0.6, 0.8], invalid = [0.0, 0.0], assort_method = "STS")
    assert np.mean(pop[0]) == 0.6
    assert np.mean(pop[1]) == 0.5
    # STS hybrid audit with invalids
    pop = generate_hybrid_audit_population(N = [200, 200], A_c = [0.6, 0.8], invalid = [0.5, 0.5], assort_method = "STS")
    assert np.mean(pop[0][pop[0] != 1/2]) == 0.6 # valid votes still have mean 0.6
    assert np.mean(pop[1]) == 0.5

    # basic stratified ONEAudit (without invalids)
    pop = generate_hybrid_audit_population(N = [200, 200], A_c = [0.6, 0.8], invalid = [0.0, 0.0], assort_method = "ONE")
    v = 2 * np.dot([0.5,0.5], [0.6, 0.8]) - 1 # global margin
    assert np.mean(pop[0]) == 1/(2 - v)
    assert np.mean(pop[1]) == 1/(2 - v)





def test_convex_uits():
    #these tests are probabilistic, they may sometimes fail (but should rarely)
    N = [10, 5]
    x_null_1 = [random_truncated_gaussian(0.5, 0.05, N[0]), random_truncated_gaussian(0.5, 0.05, N[1])]
    assert np.max(convex_uits(x_null_1, N, Allocations.round_robin, eta_0 = 0.5)[0]) < np.log(10) #there should be less than 1% chance this doesnt happen
    x_null_2 = [random_truncated_gaussian(0.2, 0.05, N[0]), random_truncated_gaussian(0.8, 0.05, N[1])]
    assert np.max(convex_uits(x_null_2, N, Allocations.round_robin, eta_0 = 0.5)[0]) < np.log(10)
    assert np.max(convex_uits(x_null_2, N, Allocations.more_to_larger_means, eta_0 = 0.5)[0]) < np.log(10)
    x_null_3 = [random_truncated_gaussian(0.4, 0.05, N[0]), random_truncated_gaussian(0.6, 0.05, N[1])]
    assert np.max(convex_uits(x_null_2, N, Allocations.round_robin, eta_0 = 0.5)[0]) < np.log(10)


    #test that it does reject eventually under alternative
    x_alt_1 = [random_truncated_gaussian(0.8, 0.05, N[0]), random_truncated_gaussian(0.8, 0.05, N[1])]
    assert np.max(convex_uits(x_alt_1, N, Allocations.round_robin, eta_0 = 0.5)[0]) > np.log(2)

    #test minimax-eta strategy (greedy kelly) under null and alternative
    assert np.max(convex_uits(x_null_1, N, Allocations.greedy_kelly, eta_0 = 0.5)[0]) < np.log(10)
    assert np.max(convex_uits(x_alt_1, N, Allocations.greedy_kelly, eta_0 = 0.5)[0]) > np.log(2)

    #check that greedy_kelly pulls different strata than round robin when the strata are different
    x_alt_2 = [random_truncated_gaussian(0.5, 0.05, N[0]), random_truncated_gaussian(0.8, 0.05, N[1])]
    uits_rr_alt2 = convex_uits(x_alt_2, N, Allocations.round_robin, eta_0 = 0.5)
    uits_minimax_alt2 = convex_uits(x_alt_2, N, Allocations.greedy_kelly, eta_0 = 0.5)
    assert all(uits_rr_alt2[2][2,:] == uits_minimax_alt2[2][2,:]) #first 2 selections should always be round robin
    assert any(uits_rr_alt2[2][30,:] != uits_minimax_alt2[2][30,:]) #but should eventually diverge...
    assert uits_minimax_alt2[0][30] > uits_rr_alt2[0][30] #check if minimax is larger

    #check PGD works for higher dimensions
    K = 5
    N = [10 for _ in range(K)]
    x_alt_1 = [random_truncated_gaussian(0.8, 0.05, N[k]) for k in range (K)]
    assert np.max(convex_uits(x_alt_1, N, Allocations.round_robin, eta_0 = 0.5)[0]) > np.log(5)
