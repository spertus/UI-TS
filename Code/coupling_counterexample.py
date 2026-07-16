"""
Counterexample: a greedy Kelly UI-TS need not minimize the expected stopping time.

Setup (Appendix: "Greedy Kelly UI-TSs need not minimize the expected stopping time"):
  K = 2 strata, w = [1/2, 1/2], eta_0 = 1/2, sampling WITH replacement.
  Both strata are Bernoulli(p) with p = 0.9.
  The UI-TS minimizes over two intersection nulls in C:
      eta_A = [0.49, 0.51],  eta_B = [0.51, 0.49].

  GREEDY KELLY UI-TS M*:
      constituent at eta_A: always draws stratum 1, Kelly bet vs. 0.49
      constituent at eta_B: always draws stratum 2, Kelly bet vs. 0.49  (symmetric)
      => tau_A, tau_B are driven by DISJOINT streams: i.i.d.

  COMPETITOR UI-TS M':
      constituent at eta_A: same as greedy (stratum 1, Kelly bet vs. 0.49)
      constituent at eta_B: ALSO draws stratum 1, Kelly bet vs. 0.51
      => tau'_A, tau'_B are driven by the SAME stream: nearly comonotone.

  M' has a strictly worse marginal at eta_B (g(0.51) < g(0.49)) but pays no
  dispersion penalty, and wins on E[max] once alpha is small enough.

Both UI-TS stopping times are tau = max(tau_A, tau_B), since a UI-TS crosses
1/alpha only when every constituent has crossed.

Usage:  python coupling_counterexample.py
Output: the LaTeX table body in the appendix, plus diagnostics.
"""

import numpy as np

P = 0.9                     # true stratum means mu*_1 = mu*_2 = p
ETA_LO, ETA_HI = 0.49, 0.51  # the two null components appearing in eta_A, eta_B
REPS = 400_000              # sample paths per alpha
CHUNK = 20_000              # paths per batch (bounds peak memory)
SEED = 42
SETTINGS = [                # (alpha, horizon); horizon caps the walk length
    (0.05, 400),
    (1e-4, 600),
    (1e-6, 900),
]


def kelly_bet(eta, p=P):
    """Kelly-optimal bet for a Bernoulli(p) stratum against null mean eta."""
    return (p - eta) / (eta * (1.0 - eta))


def log_increments(eta, p=P):
    """(up, down, g): the two possible log increments and their mean."""
    lam = kelly_bet(eta, p)
    up = np.log1p(lam * (1.0 - eta))    # X = 1
    dn = np.log1p(-lam * eta)           # X = 0
    return up, dn, p * up + (1.0 - p) * dn


def first_passage(draws, up, dn, threshold):
    """First time the log-wealth random walk reaches `threshold`.

    draws: (reps, horizon) boolean array of Bernoulli outcomes.
    Returns float array of stopping times; paths that never cross are set to
    horizon + 1 (and counted separately by the caller).
    """
    walk = np.cumsum(np.where(draws, up, dn), axis=1)
    crossed = walk >= threshold
    tau = np.argmax(crossed, axis=1) + 1        # argmax finds first True
    never = ~crossed.any(axis=1)
    tau[never] = draws.shape[1] + 1
    return tau.astype(float), int(never.sum())


def run(alpha, horizon, rng):
    """Return (E[tau] greedy, E[tau] competitor, standard errors, diagnostics)."""
    L = np.log(1.0 / alpha)
    up_lo, dn_lo, g_lo = log_increments(ETA_LO)
    up_hi, dn_hi, g_hi = log_increments(ETA_HI)

    n = 0
    s_greedy = s_greedy2 = s_comp = s_comp2 = 0.0
    s_marg = 0.0          # E[tau_A], the common greedy marginal
    censored = 0

    for _ in range(REPS // CHUNK):
        # stratum 1 and stratum 2 data streams
        x1 = rng.random((CHUNK, horizon)) < P
        x2 = rng.random((CHUNK, horizon)) < P

        # greedy: eta_A on stream 1 vs 0.49; eta_B on stream 2 vs 0.49
        tau_a, c1 = first_passage(x1, up_lo, dn_lo, L)
        tau_b, c2 = first_passage(x2, up_lo, dn_lo, L)

        # competitor: eta_A on stream 1 vs 0.49; eta_B on stream 1 vs 0.51
        tau_b_prime, c3 = first_passage(x1, up_hi, dn_hi, L)

        greedy = np.maximum(tau_a, tau_b)
        comp = np.maximum(tau_a, tau_b_prime)

        s_greedy += greedy.sum()
        s_greedy2 += (greedy ** 2).sum()
        s_comp += comp.sum()
        s_comp2 += (comp ** 2).sum()
        s_marg += tau_a.sum()
        censored += c1 + c2 + c3
        n += CHUNK

    e_greedy = s_greedy / n
    e_comp = s_comp / n
    se_greedy = np.sqrt(max(s_greedy2 / n - e_greedy ** 2, 0.0) / n)
    se_comp = np.sqrt(max(s_comp2 / n - e_comp ** 2, 0.0) / n)

    return {
        "alpha": alpha,
        "L": L,
        "g_lo": g_lo,
        "g_hi": g_hi,
        "greedy": e_greedy,
        "comp": e_comp,
        "se_greedy": se_greedy,
        "se_comp": se_comp,
        "marginal": s_marg / n,
        "censored": censored,
    }


def main():
    rng = np.random.default_rng(SEED)
    up_lo, dn_lo, g_lo = log_increments(ETA_LO)
    up_hi, dn_hi, g_hi = log_increments(ETA_HI)

    print("Bernoulli(p) strata, p =", P)
    print(f"  eta = {ETA_LO}: lambda* = {kelly_bet(ETA_LO):.4f}, "
          f"increments +{up_lo:.4f} / {dn_lo:.4f}, g = {g_lo:.4f}")
    print(f"  eta = {ETA_HI}: lambda* = {kelly_bet(ETA_HI):.4f}, "
          f"increments +{up_hi:.4f} / {dn_hi:.4f}, g = {g_hi:.4f}")
    print(f"  {REPS:,} sample paths per alpha, seed = {SEED}\n")

    results = [run(alpha, horizon, rng) for alpha, horizon in SETTINGS]

    print(f"{'alpha':>8} {'E[tau] greedy':>16} {'E[tau] comp':>16} "
          f"{'advantage':>11} {'E[tau_A]':>10} {'dispersion':>11} {'censored':>9}")
    for r in results:
        print(f"{r['alpha']:>8g} "
              f"{r['greedy']:>10.3f} +/- {2*r['se_greedy']:.3f} "
              f"{r['comp']:>10.3f} +/- {2*r['se_comp']:.3f} "
              f"{r['greedy'] - r['comp']:>11.3f} "
              f"{r['marginal']:>10.3f} "
              f"{r['greedy'] - r['marginal']:>11.3f} "
              f"{r['censored']:>9d}")

    print("\n(dispersion = E[max(tau_A, tau_B)] - E[tau_A], the penalty the")
    print(" competitor avoids by coupling its constituents.)")

    # ---- LaTeX table body ----
    cols = " & ".join(f"$\\alpha = {a:g}$" if a >= 0.01 else
                      f"$\\alpha = 10^{{{int(round(np.log10(a)))}}}$"
                      for a, _ in SETTINGS)
    greedy_row = " & ".join(f"{r['greedy']:.2f}" for r in results)
    comp_row = " & ".join(f"{r['comp']:.2f}" for r in results)
    adv_row = " & ".join(f"{r['greedy'] - r['comp']:.2f}" for r in results)
    max_se = max(max(2 * r["se_greedy"], 2 * r["se_comp"]) for r in results)

    print("\n" + "=" * 78)
    print("LaTeX table body (requires booktabs):\n")
    print("\\begin{center}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print(f" & {cols} \\\\")
    print("\\midrule")
    print(f"Greedy Kelly UI-TS $M^*_t$ & {greedy_row} \\\\")
    print(f"Competitor $M'_t$          & {comp_row} \\\\")
    print("\\midrule")
    print(f"Advantage of $M'_t$        & {adv_row} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{center}")
    print("=" * 78)
    print(f"\nLargest Monte Carlo 2*SE across cells: {max_se:.3f}")


if __name__ == "__main__":
    main()
