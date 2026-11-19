# Convexity of UI-TSs: code for the draft note

Companion code for the draft note *Betting rules that make union-of-intersections test sequences convex*
(`../manuscript/main.tex`). Notation and betting-rule conventions follow
[Spertus, Sridhar, and Stark (arXiv:2409.06680, v5)](https://arxiv.org/abs/2409.06680) and `utils.Bets` in this repository.

## Files

| File | What it does |
|---|---|
| `uits_convexity.py` | Betting rules vectorised over a grid of null means (fixed, AGRAPA, GRAPA, binary Kelly, ST-Bern/ALPHA, negative exponential); the universal-portfolio (UP) mixture computed exactly by Gauss–Legendre quadrature; a lattice on C for K = 2, 3 (`Lattice`); a sequential engine (`run_uits`) with round-robin, greedy (UCB on the Kelly growth rate at the estimated least-favourable null) and oracle selection, returning the UI-TS path `max_j min_η M_j(η)` and the stopping time; and convex envelopes. The null-free fractions of Section 4 (`kelly_fraction`, `taylor_fraction`), the greedy growth estimate (`growth_at`), and, for any K, the separable objectives `FractionObjective` (S1/S2) and `MixtureObjective` (S3) with the water-filling solver `waterfill_min` are imported from `Code/utils.py` (see below). |
| `simulations.py` | Experiments behind the note: **B** K = 3 UI-TS at fixed n (Table 2; local-search misses and convexification losses quoted in the text), **E** AGRAPA local minima for K = 2 (one sentence in Section 3.3), **F** stopping times under Beta alternatives (Figure 3), **G** timing of water-filling vs. interior point as K grows (Table 1), **H** perfect stratification of two-point populations, stratified UI-TS vs. unstratified TSM (Figure 4), **I** selection rules: round robin vs. greedy vs. oracle (Table 3). Writes CSVs to `../results/`, LaTeX tables and `numbers.tex` to `../manuscript/tables/`. |
| `figures.py` | Figure 1 (known-alternative Kelly), Figure 2 (K = 2 anatomy), Figure 3 (stopping times), Figure 4 (two-point populations). |
| `test_against_repo.py` | Checks the vectorised rules against `utils.Bets` (they agree to ~1e-12); checks that the product of the repository's predictable UP bets equals the UP mixture (this holds since the predictability fix to `Bets.universal_portfolio`; before it, the bet on observation `i` used `z[:i]`, which includes observation `i` itself); and checks water-filling against `cvxopt.solvers.cp` on a small problem. |

## Methods that live in `Code/utils.py`

The reusable methods were ported to the repository's `Code/utils.py`, with tests in `Code/tests.py`.
`uits_convexity.py` adds `Code/` to `sys.path` (relative to its own location) and imports them from there, so the
scripts still run from this directory.

| Here (before the port) | In `Code/utils.py` | Used here |
|---|---|---|
| `kelly_fraction`, `taylor_fraction` | same names and signatures | imported |
| `growth_at` | same name and signature | imported |
| `_pad`, `FractionObjective`, `MixtureObjective`, `waterfill_min` | same names and signatures | imported (`_pad` only through the objectives) |
| `log_wealth_up` (with `_quad_nodes`, `_up_terms`) | `up_log_mart(x, eta, prior)` | no; the local copy is still used |
| S1/S2 bets of `StratumBettor` | `Bets.kelly_reference`, `Bets.taylor_reference` (kwargs `eta_ref`, `c_max`, `sd_min`) | no |
| `run_uits(selection="greedy")` | `Allocations.greedy_kelly_hindsight` | no |
| the sequential engine `run_uits` for any K | `waterfill_uits(x, N, allocation_func, strategy, ...)`, which minimises by water-filling instead of over a lattice | no |

`Lattice`, `StratumBettor`, the lattice `run_uits`, the vectorised `bets_*` rules, `lower_envelope` and the grid helpers
are specific to the note's figures and stay here. One small difference: `Bets.taylor_reference` and
`waterfill_uits(strategy="taylor_ref")` set the first fraction to 0, as `kelly_fraction` does with no data, whereas
`StratumBettor` uses the Taylor fraction at the prior values `μ̂_0 = 1/2`, `σ̂_0 = 1/4` for the first draw.

## Strategy names in the code

| Code | Note |
|---|---|
| `kelly_ref` | S1: predictable Kelly fraction at the estimated least-favourable null (the current minimiser of the objective) |
| `taylor_ref` | S2: Taylor (AGRAPA-style) approximation to S1 |
| `up` | S3: universal-portfolio mixture over the fraction |
| `fixed`, `agrapa`, `grapa`, `binary_kelly`, `alpha`, `negexp` | the repository's rules |
| `selection="greedy"` in `run_uits` | Section 4.4: argmax of mean + 2·max(sd, 0.05)/√n of the per-draw log-growth of the hindsight Kelly bet at the lagged minimiser |

## Reproduce

```bash
pip install numpy scipy pandas matplotlib cvxopt pypoman iteround welford   # importing Code/utils.py needs the repository's dependencies
python simulations.py            # about 30-40 minutes on 2 cores; --reps-* flags shrink it, --only selects experiments
python figures.py
python test_against_repo.py ../../Code   # path to the repository's Code/ directory (needs its dependencies)
```

`python simulations.py --only bfhi --reps-b 10 --reps-f 20 --reps-h 10 --reps-i 10` is a quick smoke test.

## Conventions

* Sampling is with replacement and selection is η-oblivious (round robin unless stated), so `log M_t(η) = Σ_k L_k(η_k)`.
* Lagged estimates: `μ̂_0 = 1/2`, `σ̂_0 = 1/4`, sample SD with `ddof = 1` (matches `Bets.lag_welford`).
* AGRAPA uses `c = 0.75`; GRAPA is capped at `0.99/η` (the repository's `kelly_optimal` allows `1/η`); S1 and S2 cap the
  fraction at `0.9`; the UP mixture uses a uniform prior on the fraction.
* The objective is evaluated on the lattice of points of C whose coordinates are multiples of 1/200 (equal weights,
  η_0 = 1/2); the reference null for S1 and S2 is the lattice minimiser of the current I-TSM (without the running max).
* Stopping times use the UI-TS with the running maximum outside the minimum, `max_{j≤t} min_η M_j(η)` (`running_max="outer"`, the default;
  `"inner"` gives the paper's `min_η max_{j≤t} M_j(η)`).
* Water-filling: safeguarded Newton on `(-L_k')^(-1/2)` for each stratum's root and Brent's method on `log(-ν)` for the
  level; about 60-130 slope evaluations per solve, each O(total draws) for S1/S2 and O(total draws × nodes) for S3.
