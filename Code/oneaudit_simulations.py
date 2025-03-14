import scipy as sp
import pandas as pd
import itertools
import random
import numpy as np
import time
from iteround import saferound
from utils import Bets, Allocations, Weights, mart, lower_confidence_bound, global_lower_bound,\
    intersection_mart, plot_marts_eta, construct_exhaustive_eta_grid, selector,\
    construct_eta_grid_plurcomp, construct_eta_bands, simulate_plurcomp, PGD, convex_uits,\
    banded_uits, brute_force_uits

# Notes:
# for modeling San Francisco there are many batches (say 90%) of size 1
# then you would stratify on the batches that are size 1 and the batches that are bigger than 1
# smallest batch larger than 1 would be like size 100
# control the proportion of sizes that are larger than 1 (20% to 90%) where 20% is a state without much vote by mail, 90% is SF
# we could do 10, 25, 50, 75, 90
# Federal limit of 2000 registered voters per precinct;
# batches could be uniformly distributed on a grid from [100, 1000]
# if the population size is set, create batches until all the population is allocated
# or could do a stick breaking thing


A_c_global_grid = np.linspace(0.51, 0.75, 20) # global assorter means
delta_grid = [0, 0.1, 0.5] # maximum spread between batches, assuming batches are equally spread
num_batch_grid = [2, 10, 100, 1000] # the number of batches
batch_size_grid = [100, 1000] # assuming for now, equally sized batches
prop_invalid_grid = [0, 0.1, 0.5]
alpha = 0.05 # risk limit
n_reps = 1000 # the number of replicate simulations

bets_dict = {
    "agrapa":lambda x, eta: Bets.agrapa(x, eta, c = 0.9),
    "alpha": lambda x, eta: Bets.inverse_eta(x, eta, c = 0.9),
    "grapa": "TODO",
    "kelly-optimal": lambda x, eta: Bets.kelly_optimal(x, eta)}
bets_grid = list(bets_dict.keys())

results = []

for A_c_global, delta, num_batches, batch_size, prop_invalid, bet in itertools.product(A_c_global_grid, delta_grid, num_batch_grid, batch_size_grid, prop_invalid_grid, bets_grid):
    u = 1 # upper bound for plurality assorters
    v_global = 2 * A_c_global - 1 # global margin
    A_c = np.arange(A_c_global - 0.5 * delta, A_c_global - 0.5 + delta, num_batches)
    batch_sizes = np.ones(num_batches) * batch_size


    assorter_pop_unscaled = generate_oneaudit_population(batch_sizes = batch_sizes, A_c = A_c, invalid = prop_invalid)
    eta_0_unscaled = (1/2) # global null mean

    # assorters and global null are rescaled to [0,1] for compatability with functions from utils
    # NB: if stratification is used, may need to rethink rescaling: each stratum needs to be on [0,1] and the global null should still correspond to the assertion
    assorter_pop = assorter_pop_unscaled / (2 * u / (2 * u - v))
    eta_0 = eta_0_unscaled / (2 * u / (2 * u - v))
    N = len(assorter_pop) # the size of the population/sample

    #derive kelly-optimal bet by applying numerical optimization to entire population
    if bet == "kelly-optimal":
        ko_bet = bets_dict[bets_grid](assorter_pop, eta_0)

    stopping_times = np.zeros(n_reps) # container for stopping times in each simulation
    for r in range(n_reps):
        X = np.random.shuffle(assorter_pop) # the sample is a permutation of the population
        # TSMs are computed for sampling WOR
        if bet == "kelly-optimal":
            mart = mart(X, eta = eta_0, lam = ko_bet * np.ones(N), N = N, log = True)
        else:
            mart = mart(X, eta = eta_0, lam_func = bets_dict[bet], N = N, log = True)
        stopping_time = np.where(any(mart > -np.log(alpha)), np.argmax(mart > -np.log(alpha)), N) # where the TSM crosses 1/alpha, or else the population size
        stopping_times[r] = stopping_time

    expected_sample_size = np.mean(stopping_times)
    percentile_sample_size = np.percentile(stopping_times, 90)

    data_dict = {
        "alt":alt,
        "delta":delta,
        "num_batches":num_batches,
        "batch_size":batch_size,
        "prop_invalid":prop_invalid,
        "bet":str(bet),
        "expected_sample_size":expected_sample_size,
        "90th_percentile_sample_size":percentile_sample_size,
        "run_time":run_time}
    results.append(data_dict)
results = pd.DataFrame(results)
results.to_csv("one_audit_results.csv", index = False)
