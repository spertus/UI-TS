# Union-of-intersections test sequences (UI-TSs)

Code for [Spertus, Sridhar, and Stark (2024)](https://arxiv.org/abs/2409.06680).
Contains tools to run hypothesis tests about means of bounded, stratified populations with sequential, finite-sample, nonparametric (SFSNP) level control. 
The primary tools are union-of-intersections test sequences (UI-TSs) and sequentially valid lower confidence bounds (LCBs).

The `Code` directory contains the following python scripts:

- `utils.py` contains all functions implementing the proposed methods
- `test.py` contains unit test for all functions
- `significance_simulations.py` runs the t-test simulation that generated Figure 1
- `pointmass_simulations.py` runs the pointmass simulations presented in Section 7.1
- `bernoulli_simulations.py` runs the Bernoulli simulations presented in Section 7.2
- `gaussian_simulations.py` runs the Gaussian simulations presented in Section 7.3
- `pgd_time.py` times the convex UI-TS against the LCB method, as described in Section C.4
- `r_plots.R` contains the R code used to generate all plots in the paper

The `Results` directory contains CSV files with the simulation results
