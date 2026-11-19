# Convexity of UI-TSs: code for the draft note

Companion code for the draft note *Betting rules that make union-of-intersections test sequences convex*
(`../manuscript/main.tex`). Notation and betting-rule conventions follow
[Spertus, Sridhar, and Stark (arXiv:2409.06680, v5)](https://arxiv.org/abs/2409.06680) and `utils.Bets` in this repository.

## Files

| File | What it does |
|---|---|
| `uits_convexity.py` | Betting rules vectorised over a grid of null means (fixed, AGRAPA, GRAPA, binary Kelly, ST-Bern/ALPHA, negative exponential); the universal-portfolio (UP) mixture computed exactly by Gauss–Legendre quadrature; the null-free fractions of Section 4 (`kelly_fraction`, `taylor_fraction`); a lattice on C for K = 2, 3 (`Lattice`); a sequential engine (`run_uits`) with round-robin, greedy (UCB on the Kelly growth rate at the estimated least-favourable null) and oracle selection, returning the UI-TS path `max_j min_η M_j(η)` and the stopping time; convex envelopes; and, for any K, the separable objectives `FractionObjective` (S1/S2) and `MixtureObjective` (S3) with the water-filling solver `waterfill_min`. |
| `simulations.py` | Experiments behind the note: **B** K = 3 UI-TS at fixed n (Table 2; local-search misses and convexification losses quoted in the text), **E** AGRAPA local minima for K = 2 (one sentence in Section 3.3), **F** stopping times under Beta alternatives (Figure 3), **G** timing of water-filling vs. interior point as K grows (Table 1), **H** perfect stratification of two-point populations, stratified UI-TS vs. unstratified TSM (Figure 4), **I** selection rules: round robin vs. greedy vs. oracle (Table 3). Writes CSVs to `../results/`, LaTeX tables and `numbers.tex` to `../manuscript/tables/`. |
| `figures.py` | Figure 1 (known-alternative Kelly), Figure 2 (K = 2 anatomy), Figure 3 (stopping times), Figure 4 (two-point populations). |
| `test_against_repo.py` | Checks the vectorised rules against `utils.Bets` (they agree to ~1e-12); checks that the product of the repository's predictable UP bets equals the UP mixture, which **fails** until `fix_universal_portfolio_predictability.patch` is applied (`Bets.universal_portfolio` uses `z[:i]` for the bet on observation `i`, which includes observation `i` itself); and checks water-filling against `cvxopt.solvers.cp` on a small problem. |

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
pip install numpy scipy pandas matplotlib
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
