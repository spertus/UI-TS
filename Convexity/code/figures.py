"""Figures for the note.  Writes PDFs to the manuscript's figures/ directory.   Usage: python figures.py"""
import os, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy.stats import beta as Beta
import uits_convexity as uc
from simulations import prominent_local_min

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MAN = os.path.join(ROOT, "manuscript") if os.path.isdir(os.path.join(ROOT, "manuscript")) else ROOT
FIG = os.path.join(MAN, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results")
mpl.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
                     "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "lines.linewidth": 1.5, "savefig.bbox": "tight"})
# Fixed categorical order (CVD-checked).  Solid lines = convexifying bets; broken lines = not convexifying.
STYLE = {"up":         dict(color="#0072B2", ls="-", label="S3: UP mixture"),
         "kelly_ref":  dict(color="#009E73", ls="-", label=r"S1: Kelly fraction at $\hat{\eta}$"),
         "taylor_ref": dict(color="#56B4E9", ls="-", label=r"S2: Taylor fraction at $\hat{\eta}$"),
         "agrapa":     dict(color="#D55E00", ls="--", label="AGRAPA"),
         "grapa":      dict(color="#E69F00", ls="-.", label="GRAPA"),
         "fixed":      dict(color="#666666", ls=(0, (1, 1.3)), label="Fixed")}
MARK = {"up": "o", "kelly_ref": "s", "taylor_ref": "D", "agrapa": "v", "grapa": "P", "fixed": "x"}
INK = "#333333"; MUTED = "#8a8a8a"

# ---------------------------------------------------------------- Figure 1: known-F Kelly
def kelly_known(a=1.0, b=1.0, ngrid=600):
    mu = a / (a + b); eta = np.linspace(0.02, mu - 0.005, ngrid)
    ys = np.linspace(0, 1, 6001)[1:-1]; pw = Beta.pdf(ys, a, b); pw /= pw.sum()
    lam = np.empty_like(eta)
    for i, e in enumerate(eta):
        lo, hi = 0.0, 1 / e
        if np.sum(pw * (ys - e) / (1 + hi * (ys - e))) > 0:
            lam[i] = hi; continue
        for _ in range(90):
            m = (lo + hi) / 2
            lo, hi = (m, hi) if np.sum(pw * (ys - e) / (1 + m * (ys - e))) > 0 else (lo, m)
        lam[i] = (lo + hi) / 2
    growth = np.array([np.sum(pw * np.log1p(l * (ys - e))) for l, e in zip(lam, eta)])
    return mu, eta, lam, growth

def fig_kelly_known():
    mu, eta, lam, growth = kelly_known()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
    ax = axes[0]
    ax.plot(eta, eta * lam, color=INK)
    ax.set(xlabel=r"$\eta$", ylabel=r"$c^\star(\eta)=\eta\lambda^\star(\eta)$", xlim=(0, mu), ylim=(0, 1.03),
           title=r"(a) Kelly fraction, $F=\mathrm{Uniform}(0,1)$")
    ax = axes[1]
    cols = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
    for y, col in zip([0.01, 0.1, 0.3, 0.9], cols):
        f = np.log1p(lam * (y - eta)); f = f - f[-1]
        ax.plot(eta, f, color=col, lw=1.1, label=fr"$y={y}$")
        if y == 0.01:                                    # mark where this curve is concave
            d2 = np.gradient(np.gradient(f, eta), eta); cc = d2 < 0
            ax.plot(np.where(cc, eta, np.nan), np.where(cc, f, np.nan), color=col, lw=5, alpha=0.25, solid_capstyle="butt")
            j = np.where(cc)[0][len(np.where(cc)[0]) // 2]
            ax.annotate("concave", xy=(eta[j], f[j]), xytext=(eta[j] + 0.02, f[j] - 1.0), fontsize=7.5, color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.6))
    ax.plot(eta, growth - growth[-1], color=INK, lw=2.2, label=r"$\mathbb{E}_F$ (KL$_{\inf}$)")
    ax.set(xlabel=r"$\eta$", ylabel=r"$\log[1+\lambda^\star(\eta)(y-\eta)]$ (shifted)",
           xlim=(0.05, mu), ylim=(-3, 3), title="(b) Per-outcome log factor vs. its expectation")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_kelly_known.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: K = 2 anatomy
def k2_example(mus=(0.52, 0.90), n=60, search=300):
    """First seed whose AGRAPA objective along C has a prominent non-global local minimum."""
    lat = uc.Lattice(2, m=200)
    for seed in range(search):
        rng = np.random.default_rng(1000 + seed)
        data = [rng.beta(4 * mus[0], 4 * (1 - mus[0]), n), rng.beta(4 * mus[1], 4 * (1 - mus[1]), n)]
        obj = lat.objective([uc.log_wealth(d, lat.g, "agrapa") for d in data])
        if prominent_local_min(obj):
            return seed, data, lat
    raise RuntimeError("no example found")

def fig_k2_anatomy(rules=("up", "kelly_ref", "taylor_ref", "agrapa", "grapa", "fixed")):
    seed, data, lat = k2_example()
    res = {r: uc.run_uits(data, r, lat) for r in rules}
    g = lat.g; e1 = lat.points[:, 0]
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4))
    titles = [r"(a) Stratum 1 ($\mu_1=0.52$): $\log M_{1t}(\eta_1)$",
              r"(b) Stratum 2 ($\mu_2=0.90$): $\log M_{2t}(\eta_2)$"]
    for k in range(2):
        ax = axes[0, k]
        for r in rules:
            ax.plot(g, res[r]["L"][k], **STYLE[r])
        ax.axhline(0, color=MUTED, lw=0.5)
        ax.axvline(np.mean(data[k]), color=MUTED, lw=0.6, ls=":")
        ax.set(xlabel=fr"$\eta_{k + 1}$", xlim=(0, 1), title=titles[k])
    axes[0, 0].set_ylim(-45, 45); axes[0, 1].set_ylim(-25, 160)
    ax = axes[1, 0]
    for r in rules:
        obj = lat.objective(res[r]["L"]); i = int(obj.argmin())
        ax.plot(e1, obj, **STYLE[r]); ax.plot(e1[i], obj[i], "o", color=STYLE[r]["color"], ms=3.5, zorder=5)
        if r == "agrapa":
            mins = [j for j in range(len(obj)) if (j == 0 or obj[j] < obj[j - 1]) and (j == len(obj) - 1 or obj[j] < obj[j + 1])]
            j = min([j for j in mins if j != i], key=lambda j: obj[j])
            ax.annotate("AGRAPA: local minimum", xy=(e1[j], obj[j]), xytext=(0.04, -14), fontsize=7,
                        color=STYLE[r]["color"], arrowprops=dict(arrowstyle="->", color=STYLE[r]["color"], lw=0.6))
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set(xlabel=r"$\eta_1$  (with $\eta_2=1-\eta_1$)", xlim=(0, 1), ylim=(-20, 100),
           title=r"(c) Combined: $\log M_t(\boldsymbol{\eta})$ along $\mathcal{C}$")
    ax = axes[1, 1]
    for r in rules:
        b = res[r]["bettors"][0]
        if r == "up":
            frac = uc.up_fraction(np.asarray(b.x), g)
        elif r in uc.StratumBettor.REF_RULES:
            frac = np.full(len(g), b.fraction(res[r]["eta_hat"][0]))
        else:
            frac = g * b.bets()
        ax.plot(g, frac, **STYLE[r])
        if r in uc.StratumBettor.REF_RULES:                    # flat lines: label them directly with their value
            ax.plot(g[20::50], frac[20::50], marker=MARK[r], ms=3.5, ls="none", color=STYLE[r]["color"])
            ax.annotate(f"{r[:5].replace('kelly', 'S1').replace('taylo', 'S2')} = {frac[0]:.2f}",
                        xy=(0.63, frac[0]), xytext=(0.63, frac[0] + (0.04 if r == "kelly_ref" else -0.04)),
                        fontsize=7, color=INK, ha="left", va="center")
    ax.axvline(np.mean(data[0]), color=MUTED, lw=0.6, ls=":")
    ax.text(np.mean(data[0]) - 0.01, 1.0, r"$\hat\mu_1$", color=MUTED, fontsize=8, ha="right", va="top")
    ax.set(xlabel=r"$\eta_1$", xlim=(0, 1), ylim=(0, 1.03),
           title=r"(d) Stratum 1: fraction bet on the next draw, $\eta_1\lambda_{1t}(\eta_1)$")
    h, l = axes[1, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_k2_anatomy.pdf")); plt.close(fig)
    return seed, {r: (res[r]["eta_hat"], res[r]["fractions"]) for r in rules}

# ---------------------------------------------------------------- Figure 3: stopping times
def fig_stopping(rules=("up", "kelly_ref", "taylor_ref", "agrapa")):
    F = pd.read_csv(os.path.join(RES, "exp_f_stopping_beta_K2.csv"))
    ymax = F[F.rule.isin(rules)].groupby(["gap", "mu", "rule"]).tau.mean().max()   # fixed bets are left out: see the text
    gaps = sorted(F.gap.unique())
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9), sharex=True, gridspec_kw=dict(height_ratios=[1.15, 1]))
    for col, gap in enumerate(gaps):
        d = F[F.gap == gap]
        means = d.groupby(["mu", "rule"]).tau.mean().unstack("rule")
        cens = d.groupby(["mu", "rule"]).censored.mean().unstack("rule")
        ax = axes[0, col]
        for r in rules:
            ax.plot(means.index, means[r], marker=MARK[r], ms=3.5, **STYLE[r])
            if cens[r].max() > 0:
                c = cens[r] > 0
                ax.plot(means.index[c], means[r][c], marker=MARK[r], ms=7, ls="none", color=STYLE[r]["color"])
        ax.set_yscale("log"); ax.set_ylim(8, ymax * 1.3)
        ax.set_yticks([10, 20, 50, 100, 200]); ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, which="major")
        ax.set_title(fr"stratum means $\mu\mp{gap / 2:g}$" if gap else "equal stratum means")
        ax = axes[1, col]
        ax.axhline(1, color=MUTED, lw=0.7)
        for r in rules:
            if r in ("fixed", "agrapa"):
                continue
            ax.plot(means.index, means[r] / means["agrapa"], marker=MARK[r], ms=3.5, **STYLE[r])
        ax.set_ylim(0.6, 1.4); ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_xlabel(r"global mean $\mu$"); ax.set_xticks(sorted(d.mu.unique()))
    axes[0, 0].set_ylabel("expected draws to reject")
    axes[1, 0].set_ylabel("relative to AGRAPA")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_stopping_beta.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: two-point populations
def fig_two_point():
    H = pd.read_csv(os.path.join(RES, "exp_h_two_point.csv"))
    m = H.groupby(["mu", "delta", "design", "rule"]).tau.mean().unstack("design").reset_index()
    mus = sorted(m.mu.unique()); cols = dict(zip(mus, ["#6baed6", "#2171b5", "#08306b"]))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    for mu in mus:
        d = m[(m.mu == mu) & (m.rule == "up")]
        ax.plot(d.delta, d.stratified, color=cols[mu], ls="-", marker="o", ms=3)
        ax.plot(d.delta, d.unstratified, color=cols[mu], ls="--", marker="o", ms=3, mfc="white")
        ax.annotate(fr"$\mu={mu:g}$", xy=(d.delta.iloc[-1], d.unstratified.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK, va="center")
    ax.set_yscale("log"); ax.grid(axis="y", color="#e6e6e6", lw=0.5, which="major")
    ax.set_yticks([10, 20, 50, 100, 200, 500]); ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set(xlabel=r"spread $\delta$ between the two support points", ylabel="expected draws to reject",
           title="(a) UP mixture: stratified vs. unstratified", xlim=(-0.02, 0.92))
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=INK, ls="-", marker="o", ms=3), Line2D([], [], color=INK, ls="--", marker="o", ms=3, mfc="white")],
              ["stratified UI-TS (S3)", "unstratified TSM (UP)"], frameon=False, loc="upper left", fontsize=7.5)
    ax = axes[1]
    ax.axhline(1, color=MUTED, lw=0.7)
    for mu in mus:
        for rule, ls, mk in [("up", "-", "o"), ("kelly_ref", ":", "s")]:
            d = m[(m.mu == mu) & (m.rule == rule)]
            ax.plot(d.delta, d.unstratified / d.stratified, color=cols[mu], ls=ls, marker=mk, ms=3)
    ax.set_yscale("log"); ax.grid(axis="y", color="#e6e6e6", lw=0.5, which="major")
    ax.set_yticks([0.8, 1, 2, 5, 10]); ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.legend([Line2D([], [], color=INK, ls="-", marker="o", ms=3), Line2D([], [], color=INK, ls=":", marker="s", ms=3)]
              + [Line2D([], [], color=cols[mu], lw=2) for mu in mus],
              ["UP (S3 vs. UP TSM)", "Kelly (S1 vs. GRAPA TSM)"] + [fr"$\mu={mu:g}$" for mu in mus],
              frameon=False, loc="upper left", fontsize=7.5)
    ax.set(xlabel=r"spread $\delta$", ylabel="unstratified / stratified", title="(b) Ratio of expected draws",
           xlim=(-0.02, 0.82))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_two_point.pdf")); plt.close(fig)

if __name__ == "__main__":
    fig_kelly_known()
    seed, info = fig_k2_anatomy(); print("Figure 2 example seed offset:", seed)
    fig_stopping()
    fig_two_point()
