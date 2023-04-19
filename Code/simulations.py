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
A_c_list = [
[0.5, 0.5],
[0.6, 0.6],
[0.7, 0.7],
[0.8, 0.8],
[0.9, 0.9]
]

p_1_list = [
[0.00, 0.00],
[0.00, 0.01],
[0.00, 0.05],
[0.05, 0.05]
]

p_2_list = [
[0.00, 0.00],
[0.01, 0.01],
[0.05, 0.05]
]

bets_dict = {"fixed":Bets.fixed, "agrapa":Bets.agrapa, "smooth_predictable":Bets.smooth_predictable}
bets_list = ["fixed", "agrapa", "smooth_predictable"]

results = []
for A_c, p_1, p_2, bet in itertools.product(A_c_list, p_1_list, p_2_list, bets_list):
    stopping_times = simulate_comparison_audit(
        N, A_c, p_1, p_2,
        lam_func = bets_dict[bet],
        allocation_func = Allocations.round_robin,
        reps = 50,
        WOR = True)
    expected_stopping_time = np.mean(stopping_times)
    percentile_stopping_time = np.quantile(stopping_times, 0.9)
    data_dict = {
        "A_c":A_c,
        "p_1":p_1,
        "p_2":p_2,
        "bet":str(bet),
        "expected_stop":expected_stopping_time,
        "90percentile_stop":percentile_stopping_time
    }
    results.append(data_dict)

#save output
