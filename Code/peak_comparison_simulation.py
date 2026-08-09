# sanity-check comparison of our UI-TS (product combining) and the LCB method
# against PEAK (Cho, Gan, and Kallus 2024, https://arxiv.org/abs/2402.06122),
# which combines evidence across strata by averaging rather than multiplying
# (see Bets.peak and peak_uits in utils.py). Small and quick by design; not a full simulation study.
import time
import numpy as np
import pandas as pd
from utils import Bets, Allocations, global_lower_bound, banded_uits, peak_uits, construct_eta_bands

np.random.seed(0)

alpha = 0.05
eta_0 = 0.5
K = 2
GRID_RES = 50 # number of bands used by banded_uits; exact regardless of resolution (log-concave + endpoints)
PEAK_GRID_RES = 250 # PEAK's joint capital process is convex, not log-concave, so its grid-based
# minimum is only an upper bound on the true minimum (see test_peak_uits_basic); this needs to be
# much finer than GRID_RES to keep that approximation gap small (see grid-sensitivity check)
LCB_BREAKS = 200 # resolution of the LCB grid search; coarser than the default 1000 to keep this quick

agrapa_bet = lambda x, eta: Bets.agrapa(x, eta, c = 0.9)

results = []

def stopping_sample_size_mart(mart, sample_size, alpha, cap):
    return stopping_sample_size(mart > np.log(1/alpha), sample_size, cap)

def stopping_sample_size(crossed, sample_size, cap):
    if np.any(crossed):
        return sample_size[np.argmax(crossed)]
    else:
        return cap

############### point-mass simulation (error-free comparison-audit-like populations) ###############
print("Running point-mass simulation...")
pm_start = time.time()

N_pm = np.array([200, 200])
cap_pm = np.sum(N_pm)
alt_grid_pm = np.round(np.linspace(0.51, 0.8, 8), 4)
delta_grid_pm = [0, 0.2]
eta_bands_pm = construct_eta_bands(eta_0, N = N_pm, n_bands = GRID_RES)

for alt in alt_grid_pm:
    for delta in delta_grid_pm:
        means = [alt - delta/2, alt + delta/2]
        x = [means[k] * np.ones(N_pm[k]) for k in range(K)]

        t0 = time.time()
        ui_mart, _, ui_ss = banded_uits(x, N_pm, eta_bands_pm, agrapa_bet, Allocations.round_robin, log = True)
        ui_time = time.time() - t0
        ui_sample_size = stopping_sample_size_mart(ui_mart, ui_ss, alpha, cap_pm)

        t0 = time.time()
        lcb = global_lower_bound(x, N_pm, agrapa_bet, Allocations.round_robin, alpha, WOR = False, breaks = LCB_BREAKS)
        lcb_time = time.time() - t0
        lcb_sample_size = stopping_sample_size(lcb > eta_0, np.arange(len(lcb)), cap_pm)

        t0 = time.time()
        peak_mart, _, peak_ss, _ = peak_uits(x, N_pm, eta_0, Allocations.round_robin, n_grid = PEAK_GRID_RES, log = True)
        peak_time = time.time() - t0
        peak_sample_size = stopping_sample_size_mart(peak_mart, peak_ss, alpha, cap_pm)

        results.append({
            "population": "point_mass",
            "alt": alt,
            "delta": delta,
            "rep": 0,
            "ui_ts_sample_size": ui_sample_size,
            "ui_ts_time": ui_time,
            "lcb_sample_size": lcb_sample_size,
            "lcb_time": lcb_time,
            "peak_sample_size": peak_sample_size,
            "peak_time": peak_time})
        print(f"  alt={alt}, delta={delta}: UI-TS={ui_sample_size}, LCB={lcb_sample_size}, PEAK={peak_sample_size}")

print("point-mass simulation took " + str(round(time.time() - pm_start, 1)) + " seconds")


############### Bernoulli simulation (ballot-polling-audit-like populations) ###############
print("Running Bernoulli simulation...")
bern_start = time.time()

N_bern = np.array([600, 600])
cap_bern = np.sum(N_bern)
alt_bern = 0.6
reps = 100
eta_bands_bern = construct_eta_bands(eta_0, N = N_bern, n_bands = GRID_RES)

for rep in range(reps):
    means = [alt_bern, alt_bern]
    x = [np.random.binomial(1, means[k], N_bern[k]).astype(float) for k in range(K)]

    t0 = time.time()
    ui_mart, _, ui_ss = banded_uits(x, N_bern, eta_bands_bern, agrapa_bet, Allocations.round_robin, log = True)
    ui_time = time.time() - t0
    ui_sample_size = stopping_sample_size_mart(ui_mart, ui_ss, alpha, cap_bern)

    t0 = time.time()
    lcb = global_lower_bound(x, N_bern, agrapa_bet, Allocations.round_robin, alpha, WOR = False, breaks = LCB_BREAKS)
    lcb_time = time.time() - t0
    lcb_sample_size = stopping_sample_size(lcb > eta_0, np.arange(len(lcb)), cap_bern)

    t0 = time.time()
    peak_mart, _, peak_ss, _ = peak_uits(x, N_bern, eta_0, Allocations.round_robin, n_grid = PEAK_GRID_RES, log = True)
    peak_time = time.time() - t0
    peak_sample_size = stopping_sample_size_mart(peak_mart, peak_ss, alpha, cap_bern)

    results.append({
        "population": "bernoulli",
        "alt": alt_bern,
        "delta": 0,
        "rep": rep,
        "ui_ts_sample_size": ui_sample_size,
        "ui_ts_time": ui_time,
        "lcb_sample_size": lcb_sample_size,
        "lcb_time": lcb_time,
        "peak_sample_size": peak_sample_size,
        "peak_time": peak_time})
    if (rep + 1) % 10 == 0:
        print(f"  completed {rep + 1}/{reps} replicates")

print("Bernoulli simulation took " + str(round(time.time() - bern_start, 1)) + " seconds")

results = pd.DataFrame(results)
results.to_csv("peak_comparison_results.csv", index = False)

print("\nMean sample size by population:")
print(results.groupby("population")[["ui_ts_sample_size", "lcb_sample_size", "peak_sample_size"]].mean())
