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
    peak_uits, kelly_fraction, taylor_fraction, growth_at, up_log_mart, FractionObjective, MixtureObjective,\
    waterfill_min, waterfill_uits


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


def test_universal_portfolio_predictable():
    '''
    the universal portfolio bet on observation i may only use observations 1, ..., i-1 (predictability):
    changing x[i] must leave bets 1, ..., i unchanged, while (as a check that the test has teeth) it does
    change the bet on observation i+1
    '''
    rng = np.random.default_rng(2024)
    x = rng.uniform(0, 1, 12)
    for eta, step in [(0.3, 1), (0.5, 1), (0.7, 1), (0.5, 3)]:
        lam = Bets.universal_portfolio(x, eta, step = step)
        #with no past data, the posterior mean of the (uniformly weighted) fraction is 1/2
        assert np.isclose(lam[0], 0.5 / eta)
        for i in range(len(x)):
            x_new = x.copy()
            x_new[i] = (x[i] + 0.5) % 1
            lam_new = Bets.universal_portfolio(x_new, eta, step = step)
            np.testing.assert_array_equal(lam_new[:(i+1)], lam[:(i+1)])
            if step == 1 and i + 1 < len(x):
                assert lam_new[i+1] != lam[i+1]


########## water-filling UI-TS, null-free betting fractions, UP mixture, and greedy selection ##########
# (methods ported from Convexity/code/uits_convexity.py)

def _cvxopt_separable_min(obj, w, eta_0, eps = 1e-6):
    '''
    reference solver for the water-filling tests: the interior-point formulation (cvxopt.solvers.cp) of
    Convexity/code/simulations.py::_cvxopt_min, minimizing a FractionObjective over {eta : w . eta = eta_0, eps <= eta <= 1}
    returns (minimum, solver status)
    '''
    import cvxopt
    K = len(w)
    def second(eta): #diagonal of the Hessian
        e = np.asarray(eta)[:, None]; C, X = obj.C, obj.X
        D = 1 - C + C * X / e; A = C * X / e ** 2
        return (2 * C * X / (e ** 3 * D) - A ** 2 / D ** 2).sum(1)
    x0 = np.full(K, eta_0 / w.sum())
    def F(x = None, z = None):
        if x is None:
            return 0, cvxopt.matrix(x0)
        e = np.array(x).ravel()
        if np.any(e <= 0):
            return None
        f = float(obj.value(e).sum()); Df = cvxopt.matrix(obj.deriv(e)).T
        if z is None:
            return f, Df
        return f, Df, cvxopt.spdiag(cvxopt.matrix(z[0] * second(e)))
    G = cvxopt.matrix(np.vstack([-np.eye(K), np.eye(K)])); h = cvxopt.matrix(np.r_[-eps * np.ones(K), np.ones(K)])
    A = cvxopt.matrix(w[None, :]); b = cvxopt.matrix([float(eta_0)])
    show_progress = cvxopt.solvers.options.get("show_progress", True)
    cvxopt.solvers.options["show_progress"] = False
    try:
        sol = cvxopt.solvers.cp(F, G, h, A = A, b = b)
    finally:
        cvxopt.solvers.options["show_progress"] = show_progress
    return float(obj.value(np.array(sol["x"]).ravel()).sum()), sol["status"]


def test_waterfill_min_matches_cvxopt():
    '''water-filling agrees with a generic interior-point solve on random problems with K = 5 to 20 and no flat strata'''
    rng = np.random.default_rng(20261119)
    for K in [5, 10, 20]:
        for rep in range(2):
            w = rng.dirichlet(np.full(K, 5.0))
            xs = [rng.beta(2.4, 1.6, 30) for _ in range(K)]
            cs = [rng.uniform(0.1, 0.8, 30) for _ in range(K)]
            obj = FractionObjective(xs, cs)
            v_wf, eta_wf = waterfill_min(obj, w, 0.5)
            v_cv, status = _cvxopt_separable_min(obj, w, 0.5)
            assert status == "optimal"
            assert abs(v_wf - v_cv) < 1e-4
            assert v_wf <= v_cv + 1e-6 #the exact minimizer is no worse than the interior-point solution


def test_waterfill_min_matches_grid_K2():
    '''for K = 2, C is a segment: water-filling matches a brute-force minimum over a fine grid on it'''
    rng = np.random.default_rng(11)
    eps = 1e-6
    for rep in range(3):
        w = rng.dirichlet([3.0, 3.0])
        eta_0 = rng.uniform(0.35, 0.65)
        xs = [rng.beta(2.0, 1.5, 25), rng.beta(1.5, 1.5, 25)]
        cs = [rng.uniform(0.1, 0.8, 25) for _ in range(2)]
        lo = max(eps, (eta_0 - w[1]) / w[0])
        hi = min(1.0, (eta_0 - w[1] * eps) / w[0])
        eta_1 = np.linspace(lo, hi, 20001)
        eta_2 = (eta_0 - w[0] * eta_1) / w[1]
        #within-stratum log-TSMs on the grid, computed directly
        L_fraction = lambda k, e: np.log(1 - cs[k][None, :] + cs[k][None, :] * xs[k][None, :] / e[:, None]).sum(1)
        grid_values = {
            "fraction": L_fraction(0, eta_1) + L_fraction(1, eta_2),
            "mixture": up_log_mart(xs[0], eta_1) + up_log_mart(xs[1], eta_2)}
        for name, obj in [("fraction", FractionObjective(xs, cs)), ("mixture", MixtureObjective(xs))]:
            v_wf, eta_wf = waterfill_min(obj, w, eta_0, eps = eps)
            v_grid = np.min(grid_values[name])
            assert v_wf <= v_grid + 1e-10
            assert v_grid - v_wf < 1e-5


def test_separable_objectives_derivatives():
    '''the derivatives of FractionObjective and MixtureObjective match central differences, and show convexity'''
    rng = np.random.default_rng(5)
    xs = [rng.beta(2, 2, 20), rng.beta(3, 1, 12), rng.uniform(0, 1, 5)]
    cs = [rng.uniform(0, 0.9, 20), rng.uniform(0, 0.9, 12), rng.uniform(0, 0.9, 5)]
    eta = np.array([0.3, 0.55, 0.8])
    h = 1e-6
    for obj in [FractionObjective(xs, cs), MixtureObjective(xs)]:
        d1, d2 = obj.d12(eta)
        np.testing.assert_allclose(d1, obj.deriv(eta), rtol = 1e-10)
        np.testing.assert_allclose(d2, obj.deriv2(eta), rtol = 1e-10)
        np.testing.assert_allclose(d1, (obj.value(eta + h) - obj.value(eta - h)) / (2 * h), rtol = 1e-6)
        np.testing.assert_allclose(d2, (obj.deriv(eta + h) - obj.deriv(eta - h)) / (2 * h), rtol = 1e-6)
        assert np.all(d1 <= 0) and np.all(d2 >= 0) #nonincreasing and convex


def test_waterfill_min_feasible_and_kkt():
    '''
    the water-filling solution is feasible and satisfies the KKT conditions: interior strata have equal
    L_k'(eta_k) / w_k (the multiplier nu), and strata at eta_k = 1 have L_k'(1) / w_k <= nu.
    In the first problem, two strata have so much evidence that they sit at eta_k = 1
    '''
    eps = 1e-6
    def check(obj, w, eta_0, n_at_one = None):
        v, eta = waterfill_min(obj, w, eta_0, eps = eps)
        assert abs(w @ eta - eta_0) < 1e-10
        assert np.all(eta >= eps) and np.all(eta <= 1)
        assert np.isclose(v, np.sum(obj.value(eta)))
        at_one = eta >= 1 - 1e-12
        interior = (eta > eps) & ~at_one
        if n_at_one is not None:
            assert at_one.sum() == n_at_one
        assert interior.sum() >= 2
        slopes = obj.deriv(eta) / w
        nu = np.mean(slopes[interior])
        np.testing.assert_allclose(slopes[interior], nu, rtol = 1e-6)
        assert np.all(obj.deriv(np.ones(len(w)))[at_one] / w[at_one] <= nu)

    rng = np.random.default_rng(3)
    K = 8
    xs = [rng.beta(20, 1, 40) for _ in range(2)] + [rng.beta(1.5, 3.5, 40) for _ in range(6)]
    cs = [np.full(40, 0.9) for _ in range(2)] + [rng.uniform(0.01, 0.1, 40) for _ in range(6)]
    for obj in [FractionObjective(xs, cs), MixtureObjective(xs)]:
        check(obj, np.full(K, 1/K), 0.6, n_at_one = 2)
    #random problems with unequal weights
    for K in [3, 12]:
        w = rng.dirichlet(np.full(K, 2.0))
        xs = [rng.beta(4 * m, 4 * (1 - m), 30) for m in rng.uniform(0.3, 0.9, K)]
        cs = [rng.uniform(0.05, 0.9, 30) for _ in range(K)]
        for obj in [FractionObjective(xs, cs), MixtureObjective(xs)]:
            check(obj, w, 0.5)


def test_waterfill_min_flat_strata():
    '''
    strata whose fractions are all zero (they never bet) and unsampled strata are flat: they sit at eps, leaving the
    most null mass for the informative strata, unless the informative strata all reach eta_k = 1, in which case the flat
    strata take up the remaining null mass. All-flat problems are handled too
    '''
    eps = 1e-6
    rng = np.random.default_rng(8)
    K = 6
    w = rng.dirichlet(np.full(K, 5.0))
    xs = [rng.beta(2.4, 1.6, 30) for _ in range(K)]
    cs = [rng.uniform(0.1, 0.8, 30) for _ in range(K)]
    cs[1] = np.zeros(30) #never bets
    xs[4] = np.array([]); cs[4] = np.array([]) #unsampled
    problems = [(FractionObjective(xs, cs), FractionObjective, (xs, cs), np.array([False, True, False, False, True, False])),
                (MixtureObjective(xs), MixtureObjective, (xs,), np.array([False, False, False, False, True, False]))]
    for obj, cls, args, flat in problems:
        v, eta = waterfill_min(obj, w, 0.5, eps = eps)
        assert abs(w @ eta - 0.5) < 1e-10 and np.all(eta >= eps) and np.all(eta <= 1)
        assert np.all(eta[flat] == eps)
        #same as the problem without the flat strata, with their (minimal) null mass taken out of eta_0
        reduced = cls(*[[a[k] for k in range(K) if not flat[k]] for a in args])
        v_reduced, eta_reduced = waterfill_min(reduced, w[~flat], 0.5 - eps * w[flat].sum(), eps = eps)
        assert np.isclose(v, v_reduced + np.sum(obj.value(eta)[flat]), rtol = 0, atol = 1e-10)
        np.testing.assert_allclose(eta[~flat], eta_reduced, rtol = 0, atol = 1e-8)

    #the informative stratum has little weight, so it sits at eta = 1 and the flat strata absorb the rest
    w = np.array([0.1, 0.45, 0.45])
    x_0 = rng.beta(2.4, 1.6, 30)
    c_0 = rng.uniform(0.1, 0.8, 30)
    for obj in [FractionObjective([x_0, rng.uniform(0, 1, 30), np.array([])], [c_0, np.zeros(30), np.array([])]),
                MixtureObjective([x_0, np.array([]), np.array([])])]:
        v, eta = waterfill_min(obj, w, 0.5, eps = eps)
        assert eta[0] == 1 and abs(w @ eta - 0.5) < 1e-10
        np.testing.assert_allclose(eta[1:], 0.4 / 0.9, atol = 1e-12)
        assert np.isclose(v, obj.value(np.ones(3))[0], rtol = 0, atol = 1e-12)

    #every stratum flat
    for obj in [FractionObjective([rng.uniform(0, 1, 10), np.array([])], [np.zeros(10), np.array([])]),
                MixtureObjective([np.array([]), np.array([])])]:
        v, eta = waterfill_min(obj, np.array([0.3, 0.7]), 0.5)
        assert abs(v) < 1e-12
        np.testing.assert_allclose(eta, 0.5)


def test_waterfill_min_large_K():
    '''a K = 500 problem with 50 draws per stratum (S2 fractions at eta_0, as in the note's timing experiment) solves in under 2 s'''
    import time
    rng = np.random.default_rng(51)
    K, n_k = 500, 50
    w = rng.dirichlet(np.full(K, 5.0))
    xs = [rng.beta(4 * m, 4 * (1 - m), n_k) for m in rng.uniform(0.5, 0.8, K)]
    cs = [0.5 * Bets.taylor_reference(x, 0.5) for x in xs] #fractions c = eta * lambda
    obj = FractionObjective(xs, cs)
    start = time.perf_counter()
    v, eta = waterfill_min(obj, w, 0.5)
    elapsed = time.perf_counter() - start
    assert elapsed < 2
    assert abs(w @ eta - 0.5) < 1e-10 and np.all(eta >= 1e-6) and np.all(eta <= 1)


def test_up_mixture_matches_numerical_integration():
    '''MixtureObjective and up_log_mart agree with direct numerical integration (scipy.integrate.quad) of the UP mixture'''
    rng = np.random.default_rng(13)
    etas = np.array([0.2, 0.5, 0.9])
    for n in [1, 5, 15]:
        x = rng.uniform(0, 1, n)
        for eta in etas:
            integrand = lambda c: np.prod(1 - c + c * x / eta)
            uniform = np.log(sp.integrate.quad(integrand, 0, 1, epsabs = 0, epsrel = 1e-12)[0])
            #Beta(1/2, 1/2) prior: weight c^(-1/2) (1 - c)^(-1/2), normalized by B(1/2, 1/2) = pi
            jeffreys = np.log(sp.integrate.quad(integrand, 0, 1, weight = "alg", wvar = (-0.5, -0.5), epsabs = 0, epsrel = 1e-12)[0] / np.pi)
            assert abs(up_log_mart(x, eta) - uniform) < 1e-9
            assert abs(up_log_mart(x, eta, prior = "jeffreys") - jeffreys) < 1e-9
            assert abs(MixtureObjective([x]).value(np.array([eta]))[0] - uniform) < 1e-9
        #vectorized over eta
        np.testing.assert_allclose(up_log_mart(x, etas), [up_log_mart(x, eta) for eta in etas], rtol = 1e-12)
    assert up_log_mart(np.array([]), 0.5) == 0
    with pytest.raises(NotImplementedError):
        up_log_mart(x, 0.5, N = 100)


def test_up_log_mart_matches_universal_portfolio_bets():
    '''
    the product of the predictable Bets.universal_portfolio bets equals the UP mixture, exp(up_log_mart)
    (this needs the predictability fix: bets that use observation i for the bet on observation i overshoot)
    '''
    rng = np.random.default_rng(0)
    x = rng.beta(2.6, 1.4, 25)
    for eta in [0.3, 0.5, 0.7]:
        lam = Bets.universal_portfolio(x, eta)
        np.testing.assert_allclose(np.prod(1 + lam * (x - eta)), np.exp(up_log_mart(x, eta)), rtol = 1e-6)
        assert abs(mart(x, eta, lam_func = Bets.universal_portfolio)[-1] - up_log_mart(x, eta)) < 1e-6


def test_reference_bets_predictable():
    '''
    the first bet of Bets.kelly_reference and Bets.taylor_reference is 0, bet i is unchanged when x[i:] changes,
    and the fractions eta * lambda are in [0, c_max]
    '''
    rng = np.random.default_rng(17)
    x = rng.beta(3, 2, 20)
    eta_ref = rng.uniform(0.3, 0.6, 20)
    for bet in [Bets.kelly_reference, Bets.taylor_reference]:
        for kwargs in [{}, {"eta_ref": 0.45}, {"eta_ref": eta_ref}, {"eta_ref": eta_ref, "c_max": 0.5}]:
            lam = bet(x, 0.5, **kwargs)
            assert lam.shape == x.shape and lam[0] == 0
            assert np.all(lam >= 0) and np.all(0.5 * lam <= kwargs.get("c_max", 0.9))
            for i in range(1, len(x)):
                x_new = x.copy()
                x_new[i:] = rng.uniform(0, 1, len(x) - i)
                np.testing.assert_array_equal(bet(x_new, 0.5, **kwargs)[:(i+1)], lam[:(i+1)])
        assert len(bet(np.array([]), 0.5)) == 0
        with pytest.raises(ValueError):
            bet(x, 0.5, eta_ref = eta_ref[:-1])
        #works with mart like any other betting function
        assert mart(x, 0.5, lam_func = bet)[-1] > 0


def test_reference_bets_null_free():
    '''eta * lambda does not depend on eta when the reference nulls are given: the fractions are null-free'''
    rng = np.random.default_rng(19)
    x = rng.beta(3, 2, 25)
    etas = np.linspace(0.1, 0.95, 18)
    for bet in [Bets.kelly_reference, Bets.taylor_reference]:
        for eta_ref in [0.5, rng.uniform(0.3, 0.7, 25)]:
            fractions = np.array([eta * bet(x, eta, eta_ref = eta_ref) for eta in etas])
            np.testing.assert_allclose(fractions, np.broadcast_to(fractions[0], fractions.shape), rtol = 1e-12, atol = 0)
        #by contrast, with the default reference null (eta itself) the fractions do depend on eta
        fractions = np.array([eta * bet(x, eta) for eta in etas])
        assert not np.allclose(fractions, fractions[0])


def test_kelly_fraction():
    '''kelly_fraction matches a grid argmax of the past log-growth at the reference null, and respects c_max'''
    rng = np.random.default_rng(23)
    grid = np.linspace(0, 1, 100001)
    for c_max in [0.9, 0.5]:
        for eta_ref in [0.3, 0.5, 0.7]:
            x = rng.beta(2 * (eta_ref + 0.08), 2 * (0.92 - eta_ref), 30)
            c_grid = grid[grid <= c_max]
            growth = np.log(1 - c_grid[:, None] + c_grid[:, None] * x[None, :] / eta_ref).sum(1)
            c = kelly_fraction(x, eta_ref, c_max)
            assert 0 <= c <= c_max
            assert abs(c - c_grid[np.argmax(growth)]) <= 2e-5
            assert np.sum(np.log(1 - c + c * x / eta_ref)) >= growth.max() - 1e-12
    #the cap binds when the data are far above the reference null; the fraction is 0 when they are below it or absent
    assert kelly_fraction(np.ones(10), 0.5, c_max = 0.9) == 0.9
    assert kelly_fraction(np.ones(10), 0.5, c_max = 0.6) == 0.6
    assert kelly_fraction(np.full(10, 0.2), 0.5) == 0
    assert kelly_fraction(np.array([]), 0.5) == 0


def test_taylor_fraction_matches_agrapa():
    '''taylor_fraction(mu, sd, eta_ref) is eta_ref times the AGRAPA bet at eta_ref, wherever neither cap binds'''
    rng = np.random.default_rng(29)
    n_checked = 0
    for eta_ref in [0.3, 0.5, 0.7]:
        x = rng.beta(4 * (eta_ref + 0.05), 4 * (0.95 - eta_ref), 60)
        lag_mu, lag_sd = Bets.lag_welford(x)
        agrapa = Bets.agrapa(x, eta_ref) #defaults: c = 0.75, sd_min = 0.01
        taylor = np.array([taylor_fraction(m, s, eta_ref) for m, s in zip(lag_mu, lag_sd)]) #defaults: c_max = 0.9, sd_min = 0.01
        neither = (agrapa > 0) & (agrapa < 0.75 / (eta_ref + 1e-5)) & (taylor > 0) & (taylor < 0.9)
        np.testing.assert_allclose(taylor[neither], eta_ref * agrapa[neither], rtol = 1e-12)
        #so Bets.taylor_reference at eta = eta_ref is the AGRAPA bet there (after the first draw, whose bet is 0)
        neither[0] = False
        np.testing.assert_allclose(Bets.taylor_reference(x, eta_ref)[neither], agrapa[neither], rtol = 1e-12)
        n_checked += neither.sum()
    assert n_checked > 20


def test_convexifying_bets_log_convex():
    '''
    log mart(x, eta) has nonnegative second differences in eta for the null-free fractions (Bets.kelly_reference and
    Bets.taylor_reference with given reference nulls) and for the UP mixture. AGRAPA's does not, which is why it is not
    used with water-filling
    '''
    grid = np.linspace(0.06, 0.99, 94)
    agrapa_min = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        x = rng.beta(2.4, 1.6, 30)
        eta_ref = rng.uniform(0.3, 0.7, 30)
        curves = {
            "kelly_reference": [mart(x, e, lam_func = lambda x_, e_: Bets.kelly_reference(x_, e_, eta_ref = eta_ref))[-1] for e in grid],
            "taylor_reference": [mart(x, e, lam_func = lambda x_, e_: Bets.taylor_reference(x_, e_, eta_ref = 0.5))[-1] for e in grid],
            "up": up_log_mart(x, grid)}
        for name, log_mart in curves.items():
            assert np.min(np.diff(log_mart, 2)) >= -1e-9, name
        agrapa_min.append(np.min(np.diff([mart(x, e, lam_func = Bets.agrapa)[-1] for e in grid], 2)))
    assert min(agrapa_min) < -1e-9


def test_greedy_kelly_hindsight():
    '''
    greedy_kelly_hindsight returns a valid index, uses round robin until every stratum has 2 draws, never picks an
    exhausted stratum, and picks the clearly better stratum
    '''
    rng = np.random.default_rng(31)
    x = [rng.beta(9, 1, 20), rng.beta(5, 5, 20), rng.beta(1, 9, 20)] #stratum 0 is clearly best at eta = 1/2
    n = np.array([20, 20, 20])
    eta = np.array([0.5, 0.5, 0.5])
    select = lambda T_k: Allocations.greedy_kelly_hindsight(x, np.array(T_k), n, n, eta, None)
    #warm-up
    for T_k in [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [2, 1, 1], [2, 2, 1], [0, 5, 5]]:
        assert select(T_k) == Allocations.round_robin(x, np.array(T_k), n, n, eta, None)
    #after warm-up
    assert select([2, 2, 2]) == 0 and select([5, 5, 5]) == 0
    #exhausted strata
    assert select([20, 5, 5]) != 0
    assert select([20, 20, 5]) == 2
    #a stratum exhausted with fewer than 2 draws does not prolong the warm-up
    x_short = [x[0], x[1][:1], x[2]]
    n_short = np.array([20, 1, 20])
    assert Allocations.greedy_kelly_hindsight(x_short, np.array([3, 1, 3]), n_short, n_short, eta, None) == 0
    #valid indices
    for _ in range(20):
        T_k = rng.integers(0, 21, 3)
        if np.all(T_k == n):
            continue
        k = select(T_k)
        assert k in [0, 1, 2] and T_k[k] < n[k]


def test_waterfill_uits_matches_convex_uits():
    '''
    with inverse bets, round robin, and no running maximum, waterfill_uits reproduces convex_uits (which minimizes with
    an interior-point solver). Time alignment: convex_uits returns 0 until every stratum has been sampled (t < K with
    round robin) and stops one step early (its last entry is for time sum(n_k) - 1), so the paths are compared for
    K <= t <= sum(n_k) - 1
    '''
    rng = np.random.default_rng(37)
    N = np.array([12, 8, 6])
    x = [rng.beta(2.6, 1.4, 8), rng.beta(2.0, 2.0, 8), rng.beta(3.0, 1.2, 8)]
    K, n_total = 3, 24
    uits_cvx, eta_cvx, T_cvx = convex_uits(x, N, Allocations.round_robin, eta_0 = 0.5)
    uits_wf, eta_wf, T_wf = waterfill_uits(x, N, Allocations.round_robin, strategy = "inverse", running_max = "none")
    assert len(uits_cvx) == n_total and len(uits_wf) == n_total + 1
    assert np.all(uits_cvx[:K] == 0)
    np.testing.assert_array_equal(T_wf[:n_total], T_cvx[:n_total])
    np.testing.assert_allclose(uits_wf[K:n_total], uits_cvx[K:n_total], rtol = 0, atol = 1e-5)


def test_waterfill_uits_strategies_and_selection():
    '''
    every strategy runs with round robin and greedy selection; with running_max = "outer" the path is nondecreasing;
    the minimizers are in C; and the raw minimum equals the I-TSM at the minimizer recomputed with mart and the Bets
    functions, using the reference nulls that waterfill_uits fixed for each draw
    '''
    rng = np.random.default_rng(41)
    N = np.array([10, 20, 30])
    w = N / np.sum(N)
    x = [rng.beta(2.6, 1.4, 10), rng.beta(2.0, 2.0, 10), rng.beta(3.0, 1.2, 10)]
    n_total = 30
    for strategy in ["inverse", "kelly_ref", "taylor_ref", "up"]:
        for allocation in [Allocations.round_robin, Allocations.greedy_kelly_hindsight]:
            uits, eta_stars, T_k = waterfill_uits(x, N, allocation, strategy = strategy)
            assert uits.shape == (n_total + 1,) and eta_stars.shape == (n_total + 1, 3) and T_k.shape == (n_total + 1, 3)
            assert uits[0] == 0 and np.all(np.diff(uits) >= 0)
            np.testing.assert_array_equal(T_k[-1], [10, 10, 10])
            assert np.all(np.diff(T_k.sum(1)) == 1)
            np.testing.assert_allclose(eta_stars @ w, 0.5, rtol = 0, atol = 1e-10)
            assert np.all(eta_stars > 0) and np.all(eta_stars <= 1)

        raw, eta_stars, T_k = waterfill_uits(x, N, Allocations.round_robin, strategy = strategy, running_max = "none")
        eta_final = eta_stars[-1]
        log_mart = 0
        for k in range(3):
            #the reference null for draw j is the minimizer just before the draw was selected
            selected_at = np.searchsorted(T_k[:,k], np.arange(1, 11))
            eta_ref = eta_stars[selected_at - 1, k]
            if strategy == "inverse":
                lam = Bets.inverse_eta(x[k], eta_final[k])
            elif strategy == "kelly_ref":
                lam = Bets.kelly_reference(x[k], eta_final[k], eta_ref = eta_ref)
            elif strategy == "taylor_ref":
                lam = Bets.taylor_reference(x[k], eta_final[k], eta_ref = eta_ref)
            else:
                lam = Bets.universal_portfolio(x[k], eta_final[k])
            log_mart += mart(x[k], eta_final[k], lam = lam)[-1]
        assert abs(raw[-1] - log_mart) < 1e-6

    #log scale and assertions
    tiny = [x_k[:3] for x_k in x]
    np.testing.assert_allclose(waterfill_uits(tiny, N, Allocations.round_robin, log = False)[0],
        np.exp(waterfill_uits(tiny, N, Allocations.round_robin)[0]))
    with pytest.raises(NotImplementedError):
        waterfill_uits(tiny, N, Allocations.round_robin, WOR = True)
    with pytest.raises(AssertionError):
        waterfill_uits(tiny, N, Allocations.predictable_kelly)


def test_aoas_submission_manifest():
    '''
    AoAS_submission/ is a sealed copy of the code and results behind the AoAS submission: every file must match its
    SHA-256 hash in AoAS_submission/MANIFEST.sha256, and no file may be added or removed
    (Python and OS cruft, such as __pycache__/ and .DS_Store, is ignored)
    '''
    import hashlib, os
    sealed = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "AoAS_submission")
    manifest = {}
    with open(os.path.join(sealed, "MANIFEST.sha256")) as f:
        for line in f:
            if line.strip():
                digest, path = line.rstrip("\n").split("  ", 1)
                manifest[path] = digest
    found = []
    for root, dirs, files in os.walk(sealed):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".ipynb_checkpoints")]
        for name in files:
            path = os.path.relpath(os.path.join(root, name), sealed).replace(os.sep, "/")
            if path != "MANIFEST.sha256" and name not in (".DS_Store", ".Rhistory") and not name.endswith(".pyc"):
                found.append(path)
    assert sorted(found) == sorted(manifest), "files added to or removed from AoAS_submission/"
    changed = []
    for path, digest in manifest.items():
        with open(os.path.join(sealed, path), "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != digest:
                changed.append(path)
    assert changed == [], "sealed files changed: " + ", ".join(changed)
