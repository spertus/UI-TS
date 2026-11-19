# Union-of-intersections test sequences (UI-TSs)

Code for [Spertus, Sridhar, and Stark (2026+)](https://arxiv.org/abs/2409.06680).
Contains tools to run hypothesis tests about means of bounded, stratified populations with sequential, finite-sample, nonparametric (SFSNP) level control. 
The primary tools are union-of-intersections test sequences (UI-TSs) and sequentially valid lower confidence bounds (LCBs).

The `Code` directory contains the following python scripts:

- `utils.py` contains all functions implementing the proposed methods
- `tests.py` contains unit tests for all functions (run `python -m pytest tests.py` from `Code/`)
- `significance_simulations.py` runs the t-test simulation that generated Figure 1
- `pointmass_simulations.py` runs the pointmass simulations presented in Section 7.1
- `bernoulli_simulations.py` runs the Bernoulli simulations presented in Section 7.2
- `gaussian_simulations.py` runs Gaussian simulations (not currently described in the manuscript)
- `pgd_time.py` times the convex UI-TS against the LCB method, as described in Section C.4
- `r_plots.R` contains the R code used to generate all plots in the paper

The `Results` directory contains CSV files with the simulation results

## Code and results for the AoAS submission (`AoAS_submission/`)

`AoAS_submission/` is a frozen copy of the code, results and figures behind the paper as submitted to the
*Annals of Applied Statistics* ([arXiv:2409.06680v5](https://arxiv.org/abs/2409.06680v5)). It was copied, byte for
byte, from the tag `aoas-submission` (commit `3d69fce`).

- **Do not edit it.** `Code/tests.py::test_aoas_submission_manifest` checks every file against
  `AoAS_submission/MANIFEST.sha256`.
- `AoAS_submission/README.md` maps each figure and table of the submitted paper to the script and CSV that produced
  it, and explains how to rerun the scripts. The scripts import the sealed `AoAS_submission/Code/utils.py`, not
  `Code/utils.py`. Copy the directory elsewhere before rerunning anything, because the scripts write their output to
  the current directory.
- `Code/` keeps developing and may diverge from the sealed copy. For example, `Bets.universal_portfolio` in
  `Code/utils.py` is now predictable (the bet on observation i uses only observations 1, ..., i-1). The sealed copy
  keeps the earlier version, which none of the AoAS scripts use.

## Convexity note (`Convexity/`)

Code, LaTeX source and results for the draft note *Betting rules that make union-of-intersections test sequences
convex*. The note is about betting rules of the form lambda_k(eta_k) = c_k / eta_k, where the fraction c_k does not
depend on the null. These make the UI-TS objective convex in the intersection null, so it can be minimised exactly and
quickly for any number of strata.

- `Convexity/code/`: the note's simulations (`simulations.py`) and figures (`figures.py`), the engine they use
  (`uits_convexity.py`), and cross-checks against `Code/utils.py` (`test_against_repo.py`). See
  `Convexity/code/README.md`.
- `Convexity/manuscript/`: `main.tex`, `refs.bib`, and the generated figures and tables.
- `Convexity/results/`: the simulation results.

The reusable methods from the note are in `Code/utils.py`, and `uits_convexity.py` imports them from there.

### New in `Code/utils.py`

| Function | What it does |
|---|---|
| `kelly_fraction(x_past, eta_ref, c_max)` | predictable Kelly fraction at a reference null (S1 in the note) |
| `taylor_fraction(mu_hat, sd_hat, eta_ref, c_max, sd_min)` | its second-order (AGRAPA-style) approximation (S2) |
| `Bets.kelly_reference`, `Bets.taylor_reference` | bets `c_i / eta` from those fractions, for use with `mart` like any other `Bets` function; kwarg `eta_ref` (a float, or one reference null per draw) |
| `up_log_mart(x, eta, prior)` | exact log of the universal-portfolio mixture over the fraction (S3), vectorised over `eta` |
| `FractionObjective`, `MixtureObjective` | the separable log I-TSM objectives for S1/S2 and S3 |
| `waterfill_min(obj, w, eta_0)` | minimises a separable convex objective over C by water-filling; handles strata that never bet, where the interior-point solve in `convex_uits` can fail |
| `growth_at`, `Allocations.greedy_kelly_hindsight` | greedy selection: UCB on the hindsight Kelly growth rate at the previous minimising null |
| `waterfill_uits(x, N, allocation_func, strategy, ...)` | UI-TS driver with the same inputs and outputs as `convex_uits`, for strategies `"inverse"`, `"kelly_ref"`, `"taylor_ref"` and `"up"` |

Everything else in `utils.py` is unchanged, except for the predictability fix to `Bets.universal_portfolio` described
above.
