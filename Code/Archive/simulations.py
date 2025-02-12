import math
import pypoman
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import random
from scipy.stats import bernoulli, uniform, chi2
import numpy as np
import pickle
from scipy.stats.mstats import gmean
from numpy.testing import assert_allclose
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit
np.random.seed(123456789)

N = [500, 500]

grand_means = [0.51, 0.55, 0.6, 0.7]
stratum_gaps = [0.0, 0.1, 0.4]
p_1_list = [0.01, 0.02, 0.05]
p_2_list = [0.002, 0.01]

bets_dict = {"fixed":Bets.fixed, "agrapa":Bets.agrapa, "smooth_predictable":Bets.smooth_predictable}
bets_list = ["fixed", "agrapa", "smooth_predictable"]

results = []
for grand_mean, gap, p_1_1, p_1_2, p_2_1, p_2_2, bet in itertools.product(grand_means, stratum_gaps, p_1_list, p_1_list, p_2_list, p_2_list, bets_list):
    A_c = [grand_mean - 0.5*gap, grand_mean + 0.5*gap]
    p_1 = [p_1_1, p_1_2]
    p_2 = [p_2_1, p_2_2]
    stopping_times = simulate_comparison_audit(
        N, A_c, p_1, p_2,
        lam_func = bets_dict[bet],
        allocation_func = Allocations.round_robin,
        reps = 1,
        WOR = True)
    expected_stopping_time = np.mean(stopping_times)
    percentile_stopping_time = np.quantile(stopping_times, 0.9)
    data_dict = {
        "A_c":grand_mean,
        "stratum_gap":gap,
        "A_c_1":A_c[0],
        "A_c_2":A_c[1],
        "p_1_1":p_1_1,
        "p_1_2":p_1_2,
        "p_2_1":p_2_1,
        "p_2_2":p_2_2,
        "bet":str(bet),
        "expected_stop":expected_stopping_time,
        "90percentile_stop":percentile_stopping_time
    }
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv('blca_simulation_results.csv', index = False)
