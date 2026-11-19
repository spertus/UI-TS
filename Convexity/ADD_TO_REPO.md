# Adding this work to the UI-TS repository as a branch

These steps are written for a coding agent (or you) working in a local clone of
`github.com/spertus/UI-TS`.

1. Create the branch from the default branch:

   ```bash
   git checkout main && git pull
   git checkout -b convexity-note
   ```

2. Unzip `uits-convexity-repo.zip` at the repository root. It creates one new top-level folder and touches
   nothing else:

   ```
   Convexity/
     code/         uits_convexity.py, simulations.py, figures.py, test_against_repo.py, README.md
     manuscript/   main.tex, refs.bib, figures/*.pdf, tables/*.tex
     results/      exp_*.csv from the simulations in the note
   ```

   If an earlier version of `Convexity/` is already on the branch, replace the folder wholesale: several files were
   renamed or removed (old figures `fig_bets_curves.pdf`, `fig_uits_K2.pdf`; tables `tab_nonconvexity.tex`,
   `tab_stopping_K2.tex`, `tab_local_minima_K2.tex`; results `exp_a_nonconvexity.csv`, `exp_c_stopping_K2.csv`,
   `exp_d_running_max.csv`).

3. Optional but recommended: apply the predictability fix to `Code/utils.py`, as its own commit so it can be
   cherry-picked to `main`:

   ```bash
   git apply Convexity/fix_universal_portfolio_predictability.patch
   python Convexity/code/test_against_repo.py Code     # the UP lines should now match to ~1e-12
   git add Code/utils.py
   git commit -m "Make Bets.universal_portfolio predictable (bet i uses observations 1..i-1)"
   ```

   Without the fix, the bet on observation i uses observation i itself, which inflates the wealth and is not a
   valid TSM.

4. Commit the new folder and push:

   ```bash
   git add Convexity
   git commit -m "Draft note and code: betting rules that make the UI-TS convex"
   git push -u origin convexity-note
   ```

5. Re-run everything from `Convexity/code` with `python simulations.py && python figures.py`. That takes
   roughly 30-40 minutes on two cores; `--reps-*` and `--only` flags shrink it. The Overleaf project and `Convexity/manuscript` hold the
   same LaTeX; if you edit in Overleaf, copy `main.tex` back before committing.
