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
    generate_hybrid_audit_population, generate_oneaudit_population, construct_eta_bands_hybrid,\
    peak_uits


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
    assert mart(sample, eta = 0.4, lam_func = Bets.kelly_optimal, log = False)[-1] > 1
    assert mart(sample, eta = 0.4, lam_func = lambda x, eta: Bets.kelly_optimal(x, eta, pop = np.append(sample, 0)), log = False)[-1] > 1

    #test mixture
    eta_0 = 1/2
    mix_bets = [lambda x,eta: Bets.fixed(x, eta, c = b) for b in np.linspace(0.05,1/eta_0-0.05,100)]
    assert mart(sample, eta = eta_0, lam_func = mix_bets, log = True)[-1] == 0
    mix_mart_alternative = mart(sample + 0.1, eta = eta_0, lam_func = mix_bets, log = False)[-1]
    ko_mart_alternative = mart(sample + 0.1, eta = eta_0, lam_func = Bets.kelly_optimal, log = False)[-1]
    assert mix_mart_alternative < ko_mart_alternative

    # agrapa + kwargs
    agrapa = lambda x, eta: Bets.agrapa(x, eta, c = 0.9, sd_min = 0.2)
    assert mart(sample, eta = 0.5, lam_func = agrapa, log = False)[-1] == 1
    # universal portfolio + kwargs
    sample_up = np.random.normal(0.7, 0.1, 10)
    up_1 = lambda x, eta: Bets.universal_portfolio(x, eta, step = 1)
    up_5 = lambda x, eta: Bets.universal_portfolio(x, eta, step = 5)
    assert mart(sample_up, eta = 0.5, lam_func = up_1, log = False)[-1] >= 1
    assert mart(sample_up, eta = 0.5, lam_func = up_5, log = False)[-1] >= 1
    # cobra + kwargs
    A_c = 0.6
    cobra_sample = (1 / (2 - (2*A_c - 1))) * np.ones(10)
    cobra = lambda x, eta: Bets.cobra(x, eta, A_c = A_c)
    assert mart(cobra_sample, eta = 0.5, lam_func = cobra, log = False)[-1] > 1
    #WOR
    assert mart(sample, eta = 0.5, N = 15, lam_func = Bets.fixed, log = False)[-1] == 1
    assert mart(sample, eta = 0.4, N = 15, lam_func = Bets.agrapa, log = False)[-1] > 1
    assert mart(sample, eta = 0.1, N = 15, lam_func = Bets.fixed, log = False)[-1] == np.inf


def test_apriori_bernoulli_no_nan():
    '''
    regression test: an infinite bet (apriori_bernoulli at eta=0) times an exact-zero
    margin (x - eta == 0) used to compute 0 * inf = nan, which silently poisoned the
    entire cumulative martingale from that point forward instead of leaving it at 1
    '''
    apb = lambda x, eta: Bets.apriori_bernoulli(x, eta, mu_0 = 0.9)
    #data entirely consistent with eta = 0 should never move the martingale
    x_allzero = np.array([0., 0., 0.])
    m_allzero = mart(x_allzero, eta = 0.0, lam_func = apb, log = False)
    assert not np.any(np.isnan(m_allzero))
    np.testing.assert_array_equal(m_allzero, np.ones(4))

    #once a positive value appears, eta = 0 is certainly false and the mart should be infinite, not nan
    x_mixed = np.array([0., 0., 1., 1., 1.])
    m_mixed = mart(x_mixed, eta = 0.0, lam_func = apb, log = False)
    assert not np.any(np.isnan(m_mixed))
    np.testing.assert_array_equal(m_mixed, [1., 1., 1., np.inf, np.inf, np.inf])


def test_negative_exponential_eps():
    '''
    regression test for an off-by-one in the lagged-mean calculation used when eps is
    passed explicitly (previously divided by the wrong index, understating every lagged
    mean relative to Bets.lag_welford, which the default code path already relies on)
    '''
    x = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
    eps = 0.5
    lam_eps = Bets.negative_exponential(x, eta = 0.5, eps = eps)
    assert not np.any(np.isnan(lam_eps))
    lag_mu_hat, _ = Bets.lag_welford(x)
    expected_lam = np.exp(1 - ((1 - np.log(eps)) / lag_mu_hat) * 0.5)
    np.testing.assert_allclose(lam_eps, expected_lam)
    #mean (0.6) is above the null (0.5), so betting should grow the martingale
    negexp_eps = lambda x, eta: Bets.negative_exponential(x, eta, eps = eps)
    assert mart(x, eta = 0.5, lam_func = negexp_eps, log = False)[-1] > 1


def test_grapa():
    '''
    regression test for Bets.grapa: previously used x[0:i-1] (dropping the most recent
    lagged observation), treated an explicitly-passed empty `past` list as falsy/ignored,
    and assigned the length-i array returned by kelly_optimal directly into a scalar slot
    '''
    x = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
    lam = Bets.grapa(x, eta = 0.5)
    assert len(lam) == len(x)
    assert lam[0] == 0 #no data yet, so the first bet is always 0
    assert all(lam[1:] > 0) #mean (0.6) is above the null (0.5): later bets should be positive
    assert mart(x, eta = 0.5, lam_func = Bets.grapa, log = False)[-1] > 1

    #a null-true sample shouldn't be able to grow the martingale
    x_null = np.ones(5) * 0.5
    assert mart(x_null, eta = 0.5, lam_func = Bets.grapa, log = False)[-1] <= 1

    #the 'past' kwarg should extend a previously-computed sequence by exactly one predictable bet
    lam_full = Bets.grapa(x, eta = 0.5)
    lam_partial = list(Bets.grapa(x[:-1], eta = 0.5))
    lam_extended = Bets.grapa(x, eta = 0.5, past = lam_partial)
    assert len(lam_extended) == len(x)
    np.testing.assert_allclose(lam_extended, lam_full)



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
    #regression test: proportional_to_mart must fall back to round robin while any stratum
    #has <= 1 samples (previously that branch was computed but silently discarded, so the
    #martingale-based selection took over from t=1 instead of after the round-robin warmup)
    np.testing.assert_array_equal(selections[6,:], [2, 2, 2])
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
    bets = [Bets.fixed(samples[k], eta_1[k]) for k in np.arange(2)]
    selections_1 = selector(samples, N, Allocations.predictable_kelly, eta_1, bets)
    selections_2 = selector(samples, N, Allocations.predictable_kelly, eta_2, bets)
    assert selections_1[10,1] > selections_2[10,1]
    assert selections_1[10,0] < selections_2[10,0]


def test_intersection_mart():
    #null is true
    K = 3
    N = [10,10,5]
    sample = [np.ones(N[0]) * 0.5, np.ones(N[1]) * 0.5, np.ones(N[2]) * 0.5]
    lam = [Bets.fixed(x = sample[k], eta = 0.5) for k in range(len(sample))]
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam = lam, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam = lam, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "fisher")[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "product", log = False)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, theta_func = Weights.fixed, combine = "sum", log = False)[-1] == 1
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = Bets.fixed, allocation_func = Allocations.round_robin, combine = "fisher", log = False)[-1] == 1

    # test varying bets over strata and using mixtures of bets for each TSM
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = [Bets.fixed, Bets.agrapa, Bets.fixed], allocation_func = Allocations.round_robin)[-1] == 0
    assert intersection_mart(sample, N, eta = [0.5, 0.5, 0.5], lam_func = [[Bets.fixed, Bets.agrapa] for k in range(K)], allocation_func = Allocations.round_robin)[-1] == 0

    # without replacement
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

    #regression test: a boundary intersection null (eta_k = 0) combined with an infinite bet
    #(apriori_bernoulli) and an exact-zero sample value used to poison the martingale with nan
    N = [5, 5]
    sample = [np.array([0., 0., 1., 1., 1.]), np.array([1., 1., 1., 1., 1.])]
    apb = lambda x, eta: Bets.apriori_bernoulli(x, eta, mu_0 = 0.9)
    result = intersection_mart(sample, N, eta = [0, 1], lam_func = apb, allocation_func = Allocations.round_robin, combine = "product", log = False, WOR = True)
    assert not np.any(np.isnan(result))
    assert result[-1] == np.inf

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


def test_generate_oneaudit_population():
    # check w two batches
    A_c = [0.5, 0.6]
    batch_sizes = [200, 200]
    w = batch_sizes/np.sum(batch_sizes)
    v = 2 * np.dot(w, A_c) - 1
    invalid = [0.0, 0.0]
    pop = generate_oneaudit_population(batch_sizes = batch_sizes, A_c = A_c, invalid = invalid)
    assert len(pop) == 400
    assert np.round(np.mean(pop), 4) == np.round(1/(2-v), 4)
    # check what happens when there is error
    A_m = [0.5, 0.55]
    pop = generate_oneaudit_population(batch_sizes = batch_sizes, A_c = A_c, A_m = A_m)
    assert len(pop) == 400
    assert np.round(np.mean(pop), 4) <= np.round(1/(2-v), 4)
    # check what happens when the reported outcome is wrong
    A_m = [0.1, 0.1]
    pop = generate_oneaudit_population(batch_sizes = batch_sizes, A_c = A_c, A_m = A_m)
    assert len(pop) == 400
    assert np.round(np.mean(pop), 4) < np.round(1/(2-v), 4)
    X = np.random.choice(pop, size = len(pop), replace = False)
    m = mart(X, eta = 1/2, N = len(pop), lam_func = Bets.fixed, log = False)[-1]
    assert m < 1

    # when the reported votes are incorrect and the margin is smaller the martingale should be smaller as well
    A_m = [0.4, 0.55]
    pop_incorrect = generate_oneaudit_population(batch_sizes = batch_sizes, A_c = A_c, A_m = A_m)
    pop_correct = generate_oneaudit_population(batch_sizes = batch_sizes, A_c = A_c, A_m = A_c)
    X_correct = np.random.choice(pop_correct, size = len(pop), replace = False)
    X_incorrect = np.random.choice(pop_incorrect, size = len(pop), replace = False)
    m_correct = mart(X_correct, eta = 1/2, N = len(pop_correct), lam_func = Bets.fixed, log = False)[-1]
    m_incorrect = mart(X_incorrect, eta = 1/2, N = len(pop_incorrect), lam_func = Bets.fixed, log = False)[-1]
    assert m_correct > m_incorrect

def test_generate_hybrid_audit_population():
    # basic STS hybrid audit
    pop = generate_hybrid_audit_population(N = [200, 200], A_c = [0.6, 0.8], invalid = [0.0, 0.0], assort_method = "STS")
    assert np.round(np.mean(pop[0]), 4) == 0.6
    assert np.round(np.mean(pop[1]), 4) == 0.5
    # STS hybrid audit with invalids
    pop = generate_hybrid_audit_population(N = [200, 200], A_c = [0.6, 0.8], invalid = [0.5, 0.5], assort_method = "STS")
    assert np.round(np.mean(pop[0][pop[0] != 1/2]), 4) == 0.6 # check mean of valid votes
    assert np.round(np.mean(pop[1]), 4) == 0.5
    # basic stratified ONEAudit
    pop = generate_hybrid_audit_population(N = [200, 200], A_c = [0.6, 0.8], invalid = [0.0, 0.0], assort_method = "ONE")
    v = 2 * np.dot([0.5,0.5], [0.6, 0.8]) - 1 # global margin
    assert np.mean(pop[0]) == 1/(2 - v)
    assert np.mean(pop[1]) == 1/(2 - v)

def test_hybrid_audit():
    # STS formulation
    N_strat = [1000, 1000]
    A_c_strat = [0.2, 0.9]
    prop_invalid_strat = [0.0, 0.0]
    K = 2
    assorter_pop = generate_hybrid_audit_population(N_strat, A_c_strat, prop_invalid_strat, assort_method = "STS")
    etas = construct_eta_bands_hybrid(A_c_strat, N_strat, n_bands = 100, assort_method = "STS")
    X = []
    for k in range(K):
        X.append(np.random.permutation(assorter_pop[k]))
    m_sts = banded_uits(X, N = N_strat, etas = etas, lam_func = Bets.kelly_optimal, allocation_func = Allocations.proportional_round_robin, log = True)[0]
    assert m_sts[-1] > 1

    #ONE formulation
    assorter_pop = generate_hybrid_audit_population(N_strat, A_c_strat, prop_invalid_strat, assort_method = "ONE")
    etas = construct_eta_bands_hybrid(A_c_strat, N_strat, n_bands = 100, assort_method = "ONE")
    X = []
    for k in range(K):
        X.append(np.random.permutation(assorter_pop[k]))
    m_one = banded_uits(X, N = N_strat, etas = etas, lam_func = Bets.kelly_optimal, allocation_func = Allocations.proportional_round_robin, log = True)[0]
    assert m_one[-1] > 1

    # marginal election, martingales should be small
    N_strat = [100, 100]
    A_c_strat = [0.51, 0.51]
    assorter_pop = generate_hybrid_audit_population(N_strat, A_c_strat, prop_invalid_strat, assort_method = "STS")
    etas = construct_eta_bands_hybrid(A_c_strat, N_strat, n_bands = 100, assort_method = "STS")
    X = []
    for k in range(K):
        X.append(np.random.permutation(assorter_pop[k]))
    m_sts = banded_uits(X, N = N_strat, etas = etas, lam_func = Bets.kelly_optimal, allocation_func = Allocations.proportional_round_robin, log = True)[0]
    assorter_pop = generate_hybrid_audit_population(N_strat, A_c_strat, prop_invalid_strat, assort_method = "ONE")
    etas = construct_eta_bands_hybrid(A_c_strat, N_strat, n_bands = 100, assort_method = "ONE")
    X = []
    for k in range(K):
        X.append(np.random.permutation(assorter_pop[k]))
    m_one = banded_uits(X, N = N_strat, etas = etas, lam_func = Bets.kelly_optimal, allocation_func = Allocations.proportional_round_robin, log = True)[0]
    assert m_one[-1] < 5
    assert m_sts[-1] < 5




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


def test_bets_peak():
    '''
    Bets.peak implements lambda_t(eta) = (mu_hat_{t-1} - eta) / c, where mu_hat_{t-1} is a running
    mean regularized by one pseudo-observation of value mu_0, i.e.
        mu_hat_{t-1} = (mu_0 + sum of the first t-1 observations) / t;
    check this against a plain, independently-written (non-vectorized) reference loop
    '''
    x = np.array([0.6, 0.7, 0.5, 0.6, 0.55, 0.9, 0.1])
    eta = 0.4
    c = 0.26
    mu_0 = 0.5
    expected_lam = np.zeros(len(x))
    running_sum = 0.0
    for i in range(len(x)):
        mu_hat = (mu_0 + running_sum) / (i + 1)
        expected_lam[i] = (mu_hat - eta) / c
        running_sum += x[i]
    np.testing.assert_allclose(Bets.peak(x, eta, c = c, mu_0 = mu_0), expected_lam)
    #default kwargs should match c = 0.26, mu_0 = 0.5
    np.testing.assert_allclose(Bets.peak(x, eta), expected_lam)
    #the first bet only ever depends on mu_0, never the data
    assert Bets.peak(x, eta, mu_0 = 0.5)[0] == (0.5 - eta) / c
    #mu_0 = eta recovers a zero first bet (the convention suggested by the arXiv manuscript's
    #displayed Equations (4)-(5), as opposed to their published reference implementation)
    assert Bets.peak(x, eta = 0.4, mu_0 = 0.4)[0] == 0


def test_bets_peak_matches_reference_implementation():
    '''
    verify Bets.peak / mart(..., lam_func = Bets.peak) against an independent line-by-line port of
    Cho, Gan, and Kallus's own reference code (https://github.com/brianc0413/PEAK): the e_value()
    function in THR/thr_PEEK.R, combined with the mu_hat recursion used throughout that file and
    BAI/bai_PEEK.R (`mus[[h]] <- c(mus[[h]], (1/2 + sum(S_list[[h]])) / (length(S_list[[h]]) + 1))`,
    seeded at `mus <- list(1/2, ...)`).

    NB: this regularized-mean, mu_0 = 0.5 convention is what their published code actually uses to
    produce their Table 1-2 / Figure 2 results; it differs from the un-regularized, mu_0 = eta
    convention suggested by the arXiv manuscript's displayed Equations (4)-(5). Both are valid
    (Theorem 1's c >= 1/4 bound only requires mu_hat to be a predictable [0,1]-valued sequence, not
    any particular choice of it), but they are numerically different bets; we match their code here.
    '''
    def peak_reference_capital_process(S, m, c = 0.26, mu_0 = 0.5):
        K = 1.0
        running_sum = 0.0
        for i in range(len(S)):
            mu_hat = (mu_0 + running_sum) / (i + 1)
            lam = (mu_hat - m) / c
            K *= (1 + lam * (S[i] - m))
            running_sum += S[i]
        return K

    rng = np.random.default_rng(42)
    for m in [0.3, 0.5, 0.7]:
        S = rng.uniform(0, 1, 25)
        expected = peak_reference_capital_process(S, m)
        actual = mart(S, eta = m, lam_func = Bets.peak, log = False)[-1]
        np.testing.assert_allclose(actual, expected, rtol = 1e-10)


def test_peak_uits_joint_capital_process():
    '''
    verify that intersection_mart(..., lam_func = Bets.peak, combine = "sum", theta_func =
    Weights.fixed) -- what peak_uits uses internally -- reproduces PEAK's joint/averaged capital
    process E_t(m), Equations (8)-(9) of Cho, Gan, and Kallus (2024): K_t^a(m_a) is arm a's own
    capital process evaluated only on the rounds it was pulled (flat otherwise), and
    E_t(m) = (1/W) sum_a K_t^a(m_a); checked here against an independent oracle that walks a fixed,
    known interleaving by hand.
    '''
    N = [8, 8]
    x = [np.array([0.6, 0.55, 0.7, 0.5, 0.65, 0.6, 0.75, 0.5]),
         np.array([0.4, 0.45, 0.3, 0.5, 0.35, 0.4, 0.25, 0.5])]
    eta = np.array([0.45, 0.55]) #an intersection null on the boundary w.eta=0.5 for w=[1/2,1/2]
    T_k = selector(x, N, Allocations.round_robin, eta = None, lam = None)
    #recover the exact interleaving (which stratum is drawn at each global time step) from T_k
    selections = np.argmax(np.diff(T_k, axis = 0), axis = 1)

    def peak_reference_joint_capital_process(x_list, m_vec, selections, c = 0.26, mu_0 = 0.5):
        W = len(x_list)
        running_sum = np.zeros(W)
        n_seen = np.zeros(W, dtype = int)
        K_arms = np.ones(W)
        E = [1.0]
        for a in selections:
            mu_hat = (mu_0 + running_sum[a]) / (n_seen[a] + 1)
            lam = (mu_hat - m_vec[a]) / c
            x_val = x_list[a][n_seen[a]]
            K_arms[a] *= (1 + lam * (x_val - m_vec[a]))
            running_sum[a] += x_val
            n_seen[a] += 1
            E.append(np.mean(K_arms))
        return np.array(E)

    expected = peak_reference_joint_capital_process(x, eta, selections)
    #running_max = False: compare the raw (not anytime-maximized) joint process, matching the oracle
    actual = intersection_mart(x, N, eta = eta, lam_func = Bets.peak, T_k = T_k,
        combine = "sum", theta_func = Weights.fixed, log = False, running_max = False)
    np.testing.assert_allclose(actual, expected, rtol = 1e-10)


def test_peak_uits_basic():
    '''
    peak_uits minimizes PEAK's joint capital process, which is CONVEX (not log-concave) in eta
    (unlike our own product-combined I-TSMs), so unlike banded_uits its grid-based minimum is only
    an upper bound on the true continuous minimum over the null boundary: it can drift slightly
    above what the exact minimum would show, even when the null is exactly true, if the grid misses
    the true minimizer. Point-mass data exactly at eta_0 in both strata is the worst case for this
    (the true minimizer sits exactly at the boundary midpoint, and the process should stay exactly
    flat there), so it is used here to check that the drift (a) stays small and (b) shrinks as
    n_grid grows, rather than asserting it never leaves zero.
    '''
    N = [15, 15]
    sample = [np.ones(N[0]) * 0.5, np.ones(N[1]) * 0.5]
    coarse, _, _, _ = peak_uits(sample, N, eta_0 = 0.5, n_grid = 20)
    assert all(coarse < np.log(2)) #grid-approximation drift stays small
    fine, _, _, _ = peak_uits(sample, N, eta_0 = 0.5, n_grid = 200)
    assert fine[-1] < coarse[-1] #a finer grid gets closer to the true (flat) minimum
    #null is clearly false: process should eventually grow past a comfortable rejection margin
    sample = [np.ones(N[0]) * 0.9, np.ones(N[1]) * 0.9]
    mart_opt, _, ss, _ = peak_uits(sample, N, eta_0 = 0.5, n_grid = 20)
    assert mart_opt[-1] > np.log(5)
    #sample size should never exceed the total number of samples actually drawn
    assert ss[-1] <= np.sum(N)
