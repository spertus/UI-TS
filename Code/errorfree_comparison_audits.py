#get stopping times for error-free stratified comparison audits
import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_eta_grid, union_intersection_mart, selector,\
    construct_eta_grid_plurcomp, simulate_comparison_audit


N = [500, 500]

grand_means = [0.5, 0.51, 0.53, 0.55, 0.57, 0.6, 0.65, 0.7, 0.75]
stratum_gaps = [0.0, 0.1, 0.5]

bets_dict = {"fixed":Bets.fixed, "agrapa":Bets.agrapa, "smooth_predictable":Bets.smooth_predictable}
bets_list = ["fixed", "agrapa", "smooth_predictable"]

results = []
for grand_mean, gap, bet in itertools.product(grand_means, stratum_gaps, bets_list):
    A_c = [grand_mean - 0.5*gap, grand_mean + 0.5*gap]
    p_1 = [0.0, 0.0]
    p_2 = [0.0, 0.0]
    stopping_time_uinnsm = simulate_comparison_audit(
        N, A_c, p_1, p_2,
        lam_func = bets_dict[bet],
        allocation_func = Allocations.round_robin,
        method = "ui-nnsm",
        combine = "product",
        reps = 1,
        WOR = True)[0]
    stopping_time_lcb = simulate_comparison_audit(
        N, A_c, p_1, p_2,
        lam_func = bets_dict[bet],
        allocation_func = Allocations.round_robin,
        method = "lcbs",
        reps = 1,
        WOR = True)[0]
    data_dict = {
        "A_c":grand_mean,
        "stratum_gap":gap,
        "A_c_1":A_c[0],
        "A_c_2":A_c[1],
        "bet":str(bet),
        "stopping_time_uinnsm":stopping_time_uinnsm,
        "stopping_time_lcb":stopping_time_lcb
    }
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("comparison_audit_results.csv", index = False)
