"""
Simulations for the note.  Writes CSVs to ../results and LaTeX tables to the manuscript's tables/ directory.
    B. K = 3 UI-TS at a fixed sample size (Table 2; local-search misses quoted in the text)
    E. K = 2: how often AGRAPA's objective along C has a second local minimum (quoted in the text)
    F. K = 2 stopping times under Beta alternatives, round robin (Figure 3)
    G. time to minimise a convex UI-TS objective as K grows: water-filling vs interior point (Table 3)
    H. perfect stratification: stratified UI-TS vs unstratified TSM on two-point populations (Figure 4)
    I. selection rules: round robin vs greedy at the reference null vs oracle allocation (Table 4)
Usage: python simulations.py [--reps-b 100 --reps-e 100 --reps-f 1000 --reps-g 5 --reps-h 500 --reps-i 300] [--only befghi]
"""
import argparse, os, time, numpy as np, pandas as pd
import uits_convexity as uc

HERE = os.path.dirname(os.path.abspath(__file__))
# repository layout: <root>/code, <root>/manuscript, <root>/results; Overleaf layout: manuscript files at <root>
ROOT = os.path.join(HERE, "..")
MAN = os.path.join(ROOT, "manuscript") if os.path.isdir(os.path.join(ROOT, "manuscript")) else ROOT
RES = os.path.join(ROOT, "results"); TAB = os.path.join(MAN, "tables")
os.makedirs(RES, exist_ok=True); os.makedirs(TAB, exist_ok=True)
GRID = np.linspace(0.01, 0.99, 491)
LABEL = {"fixed": "Fixed", "negexp": "Neg.\\ exponential",
         "binary_kelly": "Binary Kelly", "alpha": "ST-Bern (ALPHA)", "agrapa": "AGRAPA",
         "grapa": "GRAPA", "hindsight": "Hindsight Kelly",
         "kelly_ref": "S1: Kelly fraction at $\\hat{\\boldsymbol\\eta}$",
         "taylor_ref": "S2: Taylor fraction at $\\hat{\\boldsymbol\\eta}$", "up": "S3: UP mixture"}
CONVEXIFYING = ["kelly_ref", "taylor_ref", "up"]

def draw(rng, family, mu, n):
    if family == "beta":
        return rng.beta(4 * mu, 4 * (1 - mu), n)
    if family == "bernoulli":
        return (rng.random(n) < mu).astype(float)
    raise ValueError(family)

# -------------------------------------------------------------------------------------- B
def lattice_descent(Ls, m, S, start):
    """Discrete steepest descent over the K = 3 lattice of C, moving along e_a - e_b."""
    f = lambda ix: Ls[0][ix[0] - 1] + Ls[1][ix[1] - 1] + Ls[2][ix[2] - 1]
    ix = np.array(start); fx = f(ix)
    dirs = [np.eye(3, dtype=int)[a] - np.eye(3, dtype=int)[b] for a in range(3) for b in range(3) if a != b]
    while True:
        best = None
        for d in dirs:
            for step in (1, 4, 16):
                z = ix + step * d
                if np.all(z >= 1) and np.all(z <= m):
                    fz = f(z)
                    if fz < fx - 1e-12 and (best is None or fz < best[0]):
                        best = (fz, z)
        if best is None:
            return fx
        fx, ix = best

def exp_b(reps, seed=11):
    """K = 3, w_k = 1/3, eta_0 = 1/2, round robin, fixed n_k: UI-TS value and local-search misses."""
    rng = np.random.default_rng(seed); rows = []
    MUS = np.array([0.58, 0.64, 0.72]); lat = uc.Lattice(3, m=200)
    rules = ["fixed", "kelly_ref", "taylor_ref", "up", "binary_kelly", "alpha", "agrapa", "grapa"]
    for n in [30, 100]:
        for r in range(reps):
            xs = [draw(rng, "beta", m, n) for m in MUS]
            for rule in rules:
                if rule in uc.StratumBettor.REF_RULES:
                    Ls = uc.run_uits(xs, rule, lat)["L"]
                else:
                    Ls = [uc.log_wealth(x, lat.g, rule) for x in xs]
                val = float(lat.objective(Ls).min())
                row = dict(n=n, rep=r, rule=rule, log_uits=val)
                if rule in ("grapa", "agrapa"):          # cost of convexifying (appendix)
                    envs = [uc.lower_envelope(lat.g, L) for L in Ls]
                    row["convexification_loss"] = val - float(lat.objective(envs).min())
                if rule != "fixed":
                    row["local_gap"] = lattice_descent(Ls, lat.m, lat.S, (lat.S // 3,) * 3) - val
                rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "exp_b_uits_K3.csv"), index=False)
    table_b(df)
    return df

def table_b(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(RES, "exp_b_uits_K3.csv"))
    rules = ["kelly_ref", "taylor_ref", "up", "fixed", "binary_kelly", "alpha", "agrapa", "grapa"]
    cls = {r: "convexifying" for r in CONVEXIFYING}; cls["fixed"] = "concavifying"
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{median $\log M_t$}\\", r"\cmidrule(lr){3-4}",
             r"Bet & Class & $n_k=30$ & $n_k=100$\\", r"\midrule"]
    for rule in rules:
        if rule == "fixed":
            lines.append(r"\addlinespace")
        d30, d100 = df[(df.rule == rule) & (df.n == 30)], df[(df.rule == rule) & (df.n == 100)]
        lines.append(f"{LABEL[rule]} & {cls.get(rule, 'neither')} & {d30.log_uits.median():.1f} & {d100.log_uits.median():.1f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_uits_K3.tex"), "w").write("\n".join(lines) + "\n")

# -------------------------------------------------------------------------------------- F
def _f_task(args):
    """One replication of experiment F for one design (common random numbers across rules)."""
    seed, mu, gap, rules, n_max, alpha = args
    rng = np.random.default_rng(seed); lat = uc.Lattice(2, m=200)
    mus = (mu - gap / 2, mu + gap / 2)
    data = [draw(rng, "beta", m_, n_max) for m_ in mus]
    out = []
    for rule in rules:
        res = uc.run_uits(data, rule, lat, alpha=alpha, stop=True)
        tau = res["tau"]
        out.append(dict(mu=mu, gap=gap, seed=seed, rule=rule,
                        tau=tau if tau is not None else 2 * n_max, censored=tau is None))
    return out

def exp_f(reps, seed=41, n_max=1500, alpha=0.05, workers=2):
    """K = 2, w_k = 1/2, eta_0 = 1/2, round robin, Beta strata with means mu -/+ gap/2:
    expected number of draws until the UI-TS (running maximum inside) reaches 1/alpha."""
    from multiprocessing import Pool
    rules = ["fixed", "kelly_ref", "taylor_ref", "up", "agrapa"]
    tasks = [(seed + 100000 * i + r, mu, gap, rules, n_max, alpha)
             for i, (mu, gap) in enumerate([(mu, gap) for gap in (0.0, 0.4) for mu in (0.55, 0.60, 0.65, 0.70, 0.75)])
             for r in range(reps)]
    with Pool(workers) as pool:
        rows = [row for chunk in pool.imap_unordered(_f_task, tasks, chunksize=20) for row in chunk]
    df = pd.DataFrame(rows).sort_values(["gap", "mu", "rule", "seed"])
    df.to_csv(os.path.join(RES, "exp_f_stopping_beta_K2.csv"), index=False)
    return df

# -------------------------------------------------------------------------------------- E
def prominent_local_min(obj, min_barrier=1.0):
    """Is there a non-global local minimum separated from the global one by a barrier of
    at least `min_barrier` log units?  (obj is the objective along the segment C.)"""
    mins = [j for j in range(len(obj)) if (j == 0 or obj[j] < obj[j - 1]) and (j == len(obj) - 1 or obj[j] < obj[j + 1])]
    g = int(np.argmin(obj))
    for j in mins:
        if j == g:
            continue
        lo, hi = sorted((j, g))
        if obj[lo:hi + 1].max() - obj[j] >= min_barrier:
            return True
    return False

def exp_e(reps, seed=31):
    rng = np.random.default_rng(seed); rows = []
    w = np.array([0.5, 0.5]); e1, e2 = uc.segment_C(w, 0.5, G=1201)
    keep = (e1 > 0.01) & (e1 < 0.99); e1, e2 = e1[keep], e2[keep]
    rules = ["agrapa", "grapa", "up"]
    for mus in [(0.55, 0.75), (0.52, 0.90)]:
        for n in [60, 150]:
            for r in range(reps):
                x1 = draw(rng, "beta", mus[0], n); x2 = draw(rng, "beta", mus[1], n)
                for rule in rules:
                    obj = uc.log_wealth(x1, e1, rule) + uc.log_wealth(x2, e2, rule)
                    rows.append(dict(mu1=mus[0], mu2=mus[1], n=n, rep=r, rule=rule,
                                     local_min=prominent_local_min(obj), global_at_endpoint=int(np.argmin(obj)) in (0, len(obj) - 1)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "exp_e_local_minima_K2.csv"), index=False)
    return df

# -------------------------------------------------------------------------------------- G
def _cvxopt_min(obj, w, eta_0, eps=1e-6):
    """Generic interior-point solve (cvxopt.solvers.cp, as in utils.convex_uits) of the same convex program."""
    import cvxopt
    cvxopt.solvers.options["show_progress"] = False
    K = len(w)
    def second(eta):                           # diagonal Hessian for FractionObjective
        e = np.asarray(eta)[:, None]; C, X = obj.C, obj.X
        D = 1 - C + C * X / e; A = C * X / e ** 2
        return (2 * C * X / (e ** 3 * D) - A ** 2 / D ** 2).sum(1)
    x0 = np.full(K, eta_0 / w.sum())
    def F(x=None, z=None):
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
    sol = cvxopt.solvers.cp(F, G, h, A=A, b=b)
    return float(obj.value(np.array(sol["x"]).ravel()).sum()), sol["status"]

def exp_g(reps=5, seed=51, n_k=50):
    """Time one minimisation of a convex UI-TS objective over C as the number of strata grows."""
    import timeit
    rng = np.random.default_rng(seed); rows = []
    for K in [2, 10, 50, 100, 200, 500, 1000]:
        for r in range(reps):
            w = rng.dirichlet(np.full(K, 5.0)); mus = rng.uniform(0.5, 0.8, K)
            xs = [draw(rng, "beta", m, n_k) for m in mus]
            cs = []
            for x in xs:                                     # S2 fractions at eta_0 for every draw
                m_, s_ = uc.lag_mean_sd(x)
                cs.append(np.array([uc.taylor_fraction(a, b, 0.5) for a, b in zip(m_, s_)]))
            fo = uc.FractionObjective(xs, cs)
            t_wf = min(timeit.repeat(lambda: uc.waterfill_min(fo, w, 0.5), number=1, repeat=3))
            v_wf, _ = uc.waterfill_min(fo, w, 0.5)
            row = dict(K=K, rep=r, n_total=K * n_k, time_waterfill=t_wf, value_waterfill=v_wf,
                       n_flat=int(sum(np.all(c == 0) for c in cs)))   # strata that never bet: objective flat in eta_k
            if K <= 500:
                t_cv = min(timeit.repeat(lambda: _cvxopt_min(fo, w, 0.5), number=1, repeat=2))
                v_cv, status = _cvxopt_min(fo, w, 0.5)
                row.update(time_cvxopt=t_cv, value_cvxopt=v_cv, status_cvxopt=status)
            if K <= 200:
                mo = uc.MixtureObjective(xs)
                row["time_waterfill_up"] = min(timeit.repeat(lambda: uc.waterfill_min(mo, w, 0.5), number=1, repeat=2))
            rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "exp_g_timing.csv"), index=False)
    table_g(df)
    return df

def table_g(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(RES, "exp_g_timing.csv"))
    ok = df.status_cvxopt == "optimal"
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{water-filling (ms)} & \multicolumn{2}{c}{interior point, S1/S2}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"$K$ & total draws & S1/S2 & S3 (UP) & ms & converged\\", r"\midrule"]
    for K, d in df.groupby("K"):
        up = d.time_waterfill_up.median(); cv = d.time_cvxopt[d.status_cvxopt == 'optimal'].median()   # converged runs
        conv = "--" if d.status_cvxopt.isna().all() else f"{int((d.status_cvxopt == 'optimal').sum())}/{len(d)}"
        lines.append(f"{K} & {int(d.n_total.iloc[0])} & {1000 * d.time_waterfill.median():.0f} & "
                     + ("--" if np.isnan(up) else f"{1000 * up:.0f}") + " & "
                     + ("--" if np.isnan(cv) else f"{1000 * cv:.0f}") + f" & {conv}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_timing.tex"), "w").write("\n".join(lines) + "\n")

# -------------------------------------------------------------------------------------- H
def _up_tsm_path(x, eta, m=128):
    """log of the one-sample UP mixture TSM at a single null, for every prefix of x."""
    z, wq = np.polynomial.legendre.leggauss(m); cs = (z + 1) / 2; lw = np.log(wq / 2)
    f = np.cumsum(np.log(1 - cs[None, :] + cs[None, :] * x[:, None] / eta), axis=0) + lw[None, :]
    mx = f.max(1); return mx + np.log(np.exp(f - mx[:, None]).sum(1))

def _grapa_tsm_tau(x, eta, thr, c_max=0.9):
    """First time the one-sample TSM with the predictable Kelly bet at eta (GRAPA at a single null) reaches thr."""
    L = 0.0
    for i in range(len(x)):
        c = uc.kelly_fraction(x[:i], eta, c_max)
        L += np.log(1 - c + c * x[i] / eta)
        if L >= thr:
            return i + 1
    return None

def _first_cross(path, thr):
    hit = np.nonzero(path >= thr)[0]
    return int(hit[0]) + 1 if len(hit) else None

def exp_h(reps=500, seed=61, n_max=5000, alpha=0.05):
    """Perfect stratification: K = 2 point-mass strata at mu -/+ delta/2 (w = 1/2) versus an unstratified
    TSM sampling the two-point population {mu - delta/2, mu + delta/2} with probabilities 1/2."""
    rng = np.random.default_rng(seed); rows = []; thr = np.log(1 / alpha); lat = uc.Lattice(2, m=200)
    for mu in [0.55, 0.60, 0.70]:
        for delta in np.round(np.arange(0.0, 0.81, 0.1), 2):
            lo, hi = mu - delta / 2, mu + delta / 2
            if lo < 0.05 or hi > 1.0:
                continue
            for rule in ["up", "kelly_ref"]:                       # stratified: deterministic data
                data = [np.full(n_max, lo), np.full(n_max, hi)]
                tau = uc.run_uits(data, rule, lat, alpha=alpha, stop=True)["tau"]
                rows.append(dict(mu=mu, delta=delta, design="stratified", rule=rule, rep=0,
                                 tau=tau if tau else 2 * n_max, censored=tau is None))
            for r in range(reps):                                   # unstratified: random two-point draws
                x = np.where(rng.random(2 * n_max) < 0.5, lo, hi)
                tau = _first_cross(_up_tsm_path(x, 0.5), thr)
                rows.append(dict(mu=mu, delta=delta, design="unstratified", rule="up", rep=r,
                                 tau=tau if tau else 2 * n_max, censored=tau is None))
                if r < 200:
                    tau = _grapa_tsm_tau(x[:2000], 0.5, thr)
                    rows.append(dict(mu=mu, delta=delta, design="unstratified", rule="kelly_ref", rep=r,
                                     tau=tau if tau else 2000, censored=tau is None))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "exp_h_two_point.csv"), index=False)
    return df

# -------------------------------------------------------------------------------------- I
def kelly_growth(mu, conc, eta_grid, c_max=0.9, ngrid=4001):
    """kappa(eta) = max_{c <= c_max} E log(1 - c + c X/eta) for X ~ Beta(conc mu, conc (1 - mu))."""
    from scipy.stats import beta as B
    ys = np.linspace(0, 1, ngrid)[1:-1]; pw = B.pdf(ys, conc * mu, conc * (1 - mu)); pw /= pw.sum()
    out = np.empty(len(eta_grid))
    for i, e in enumerate(eta_grid):
        z = ys / e - 1
        score = lambda c: np.sum(pw * z / (1 + c * z))
        if score(0) <= 0:
            out[i] = 0.0; continue
        lo, hi = 0.0, c_max
        if score(hi) > 0:
            c = hi
        else:
            for _ in range(60):
                mid = (lo + hi) / 2; lo, hi = (mid, hi) if score(mid) > 0 else (lo, mid)
            c = (lo + hi) / 2
        out[i] = np.sum(pw * np.log(1 - c + c * ys / e))
    return out

def oracle_allocation(design, c_max=0.9):
    """p* = argmax_p min_eta [p kappa_1(eta_1) + (1 - p) kappa_2(eta_2)] over C (K = 2, w = 1/2, eta_0 = 1/2)."""
    (m1, k1), (m2, k2) = design
    e1 = np.linspace(0.005, 0.995, 199)
    g1 = kelly_growth(m1, k1, e1, c_max); g2 = kelly_growth(m2, k2, 1 - e1, c_max)
    ps = np.linspace(0.01, 0.99, 99)
    V = np.array([np.min(p * g1 + (1 - p) * g2) for p in ps])
    return float(ps[np.argmax(V)]), float(V.max()), float(np.min(0.5 * g1 + 0.5 * g2))

DESIGNS_I = {"different means": ((0.40, 4), (0.70, 4)),
             "different spreads": ((0.55, 2), (0.55, 100)),
             "one easy stratum": ((0.20, 4), (0.95, 4))}

def _i_task(args):
    seed, name, rules, n_max, p_star = args
    rng = np.random.default_rng(seed); lat = uc.Lattice(2, m=200)
    (m1, k1), (m2, k2) = DESIGNS_I[name]
    data = [rng.beta(k1 * m1, k1 * (1 - m1), n_max), rng.beta(k2 * m2, k2 * (1 - m2), n_max)]
    out = []
    for rule in rules:
        for sel in ["round_robin", "greedy", "oracle"]:
            res = uc.run_uits(data, rule, lat, stop=True, selection=sel, proportions=[p_star, 1 - p_star])
            tau = res["tau"]
            out.append(dict(design=name, rule=rule, selection=sel, seed=seed, tau=tau if tau else 2 * n_max,
                            censored=tau is None, share1=res["counts"][0] / max(res["counts"].sum(), 1)))
    return out

def exp_i(reps=300, seed=71, n_max=1500, workers=2):
    from multiprocessing import Pool
    oracle = {name: oracle_allocation(d) for name, d in DESIGNS_I.items()}
    tasks = [(seed + 100000 * j + r, name, ["kelly_ref", "up"], n_max, oracle[name][0])
             for j, name in enumerate(DESIGNS_I) for r in range(reps)]
    with Pool(workers) as pool:
        rows = [row for chunk in pool.imap_unordered(_i_task, tasks, chunksize=10) for row in chunk]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "exp_i_selection.csv"), index=False)
    pd.DataFrame([dict(design=k, p_star=v[0], rate_oracle=v[1], rate_round_robin=v[2]) for k, v in oracle.items()]
                 ).to_csv(os.path.join(RES, "exp_i_oracle.csv"), index=False)
    table_i(df)
    return df

def table_i(df=None):
    if df is None:
        df = pd.read_csv(os.path.join(RES, "exp_i_selection.csv"))
    orc = pd.read_csv(os.path.join(RES, "exp_i_oracle.csv")).set_index("design")
    sels = [("round_robin", "round robin"), ("greedy", "greedy"), ("oracle", "oracle")]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{expected draws} & \multicolumn{3}{c}{share of draws from stratum 1}\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"$(\mu_1,\kappa_1)$, $(\mu_2,\kappa_2)$ & Bet & " + " & ".join(s[1] for s in sels) + " & " + " & ".join(s[1] for s in sels) + r"\\",
             r"\midrule"]
    for name, ((m1, k1), (m2, k2)) in DESIGNS_I.items():
        first = True
        for rule in ["kelly_ref", "up"]:
            d = df[(df.design == name) & (df.rule == rule)]
            means = [d[d.selection == s].tau.mean() for s, _ in sels]
            shares = [d[d.selection == s].share1.mean() for s, _ in sels]
            lab = f"({m1:g}, {k1}), ({m2:g}, {k2})" if first else ""
            lines.append(f"{lab} & {LABEL[rule].split(':')[0]} & " + " & ".join(f"{m:.0f}" for m in means) + " & "
                         + " & ".join(f"{s:.2f}" for s in shares) + r"\\")
            first = False
        lines.append(r"\addlinespace")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_selection.tex"), "w").write("\n".join(lines) + "\n")

# -------------------------------------------------------------------------------------- numbers
def write_numbers():
    """Key numbers quoted in the text, as LaTeX macros in tables/numbers.tex."""
    r = lambda f: pd.read_csv(os.path.join(RES, f))
    B, E, F, G, H, I = (r("exp_b_uits_K3.csv"), r("exp_e_local_minima_K2.csv"), r("exp_f_stopping_beta_K2.csv"),
                        r("exp_g_timing.csv"), r("exp_h_two_point.csv"), r("exp_i_selection.csv"))
    orc = r("exp_i_oracle.csv").set_index("design")
    e = E[(E.rule == "agrapa") & (E.mu2 == 0.9) & (E.n == 150)]
    b = lambda rule, n: B[(B.rule == rule) & (B.n == n)]
    fm = F.groupby(["gap", "mu", "rule"]).tau.mean().unstack("rule")
    ratio = lambda a, c: fm[a] / fm[c]
    g = G.groupby("K").median(numeric_only=True)
    h = H.groupby(["mu", "delta", "design", "rule"]).tau.mean().unstack("design")
    hr = (h["unstratified"] / h["stratified"]).xs("up", level="rule")
    hk = (h["unstratified"] / h["stratified"]).xs("kelly_ref", level="rule")
    i = I.groupby(["design", "rule", "selection"]).tau.mean().unstack("selection")
    gain = 1 - i["greedy"] / i["round_robin"]
    gap_or = i["greedy"] / i["oracle"] - 1
    macros = {
        "repsB": B.rep.nunique(), "repsF": F.seed.nunique() // len(fm), "repsH": H[H.design == "unstratified"].rep.nunique(),
        "repsI": I.seed.nunique() // I.design.nunique(), "repsE": E.rep.nunique(), "repsG": G.rep.nunique(),
        "grapaconvloss": f"{b('grapa', 100).convexification_loss.quantile(0.95):.1f}",
        "agrapaconvloss": f"{b('agrapa', 100).convexification_loss.quantile(0.95):.1f}",
        "agrapalocalminshare": f"{100 * e.local_min.mean():.0f}\\%",
        "agrapamissshare": f"{100 * (b('agrapa', 30).local_gap > .05).mean():.0f}\\%",
        **{f"{k}Kthree": f"{b(rule, 100).log_uits.median():.1f}" for k, rule in
           [("up", "up"), ("grapa", "grapa"), ("agrapa", "agrapa"), ("kellyref", "kelly_ref"), ("taylorref", "taylor_ref")]},
        "fkellyvsagrapamin": f"{100 * (1 - ratio('kelly_ref', 'agrapa').max()):.0f}\\%",
        "fkellyvsagrapamax": f"{100 * (1 - ratio('kelly_ref', 'agrapa').min()):.0f}\\%",
        "fupvsagrapamin": f"{100 * (1 - ratio('up', 'agrapa').max()):.0f}\\%",
        "fupvsagrapamax": f"{100 * (1 - ratio('up', 'agrapa').min()):.0f}\\%",
        "ftaylorvskellymax": f"{100 * (ratio('taylor_ref', 'kelly_ref') - 1).abs().max():.0f}\\%",
        "ffixedtau": f"{fm.loc[(0.0, 0.60), 'fixed']:.0f}", "fkellytau": f"{fm.loc[(0.0, 0.60), 'kelly_ref']:.0f}",
        "ffixedcens": f"{100 * F[(F.gap == 0.0) & (F.mu == 0.60) & (F.rule == 'fixed')].censored.mean():.0f}\\%",
        "ffixedgapcens": f"{100 * F[(F.gap == 0.4) & (F.mu == 0.60) & (F.rule == 'fixed')].censored.mean():.0f}\\%",
        "fsonebest": int((fm.drop(columns="fixed").idxmin(axis=1) == "kelly_ref").sum()), "fndesigns": len(fm),
        "timeKmax": int(G.K.max()), "timeNmax": int(G[G.K == G.K.max()].n_total.iloc[0]),
        "timewfKmax": f"{1000 * g.loc[G.K.max(), 'time_waterfill']:.0f}",
        "timewfKhundred": f"{1000 * g.loc[100, 'time_waterfill']:.0f}",
        "timecvKhundred": f"{1000 * g.loc[100, 'time_cvxopt']:.0f}",
        "timeupKhundred": f"{1000 * g.loc[100, 'time_waterfill_up']:.0f}",
        "timediffmax": f"{(G.value_waterfill - G.value_cvxopt)[G.status_cvxopt == 'optimal'].abs().max():.0e}",
        "timewfbelowcv": f"{(G.value_cvxopt - G.value_waterfill)[G.status_cvxopt != 'optimal'].min():.0f}" if (G.status_cvxopt.notna() & (G.status_cvxopt != 'optimal')).any() else "0",
        "timecvfail": int((G.status_cvxopt.notna() & (G.status_cvxopt != 'optimal')).sum()),
        "timecvruns": int(G.status_cvxopt.notna().sum()),
        "timecvfailflat": int((G.status_cvxopt.notna() & (G.status_cvxopt != 'optimal') & (G.n_flat > 0)).sum()),
        "timecvokflat": int(((G.status_cvxopt == 'optimal') & (G.n_flat > 0)).sum()),
        "timecvfailKmin": int(G[G.status_cvxopt.notna() & (G.status_cvxopt != 'optimal')].K.min()) if (G.status_cvxopt.notna() & (G.status_cvxopt != 'optimal')).any() else 0,
        "hratiozero": f"{hr.xs(0.0, level='delta').mean():.2f}",
        "hratioAup": f"{hr.loc[(0.55, 0.8)]:.1f}", "hratioBup": f"{hr.loc[(0.60, 0.8)]:.1f}",
        "hratioAkelly": f"{hk.loc[(0.55, 0.8)]:.1f}", "hratioBkelly": f"{hk.loc[(0.60, 0.8)]:.1f}",
        "hratioCmax": f"{hr.xs(0.70, level='mu').max():.2f}",
        "hstratAzero": f"{h.loc[(0.55, 0.0, 'up'), 'stratified']:.0f}", "hstratAmax": f"{h.loc[(0.55, 0.8, 'up'), 'stratified']:.0f}",
        "hunstratAmax": f"{h.loc[(0.55, 0.8, 'up'), 'unstratified']:.0f}",
        "iselgainmax": f"{100 * gain.drop('different means', level='design').max():.0f}\\%",
        "iselgainmin": f"{100 * gain.drop('different means', level='design').min():.0f}\\%",
        "iselmeansdiff": f"{100 * gain.xs('different means', level='design').abs().max():.0f}\\%",
        "iselvsoraclemax": f"{100 * gap_or.drop('different means', level='design').max():.0f}\\%",
        "iselbeatoracle": int((gap_or.drop('different means', level='design') < 0).sum()),
        **{f"pstar{j}": f"{orc.loc[name, 'p_star']:.2f}" for j, name in zip("ABC", DESIGNS_I)},
    }
    with open(os.path.join(TAB, "numbers.tex"), "w") as fh:
        fh.write("% generated by code/simulations.py -- do not edit by hand\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return macros

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps-b", type=int, default=100); ap.add_argument("--reps-e", type=int, default=100)
    ap.add_argument("--reps-f", type=int, default=1000); ap.add_argument("--reps-g", type=int, default=5)
    ap.add_argument("--reps-h", type=int, default=500); ap.add_argument("--reps-i", type=int, default=300)
    ap.add_argument("--only", default="befghi")
    a = ap.parse_args()
    for key, fn, reps in [("b", exp_b, a.reps_b), ("e", exp_e, a.reps_e), ("f", exp_f, a.reps_f),
                          ("g", exp_g, a.reps_g), ("h", exp_h, a.reps_h), ("i", exp_i, a.reps_i)]:
        if key in a.only:
            t0 = time.time(); fn(reps); print(f"experiment {key.upper()} done in {time.time() - t0:.0f}s", flush=True)
    try:
        print(write_numbers())
    except (FileNotFoundError, KeyError) as err:
        print("numbers.tex not written (run all experiments first):", err)
