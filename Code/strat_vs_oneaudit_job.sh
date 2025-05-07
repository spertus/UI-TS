#!/bin/bash
#SBATCH --job-name=strat_vs_oneaudit_simulations
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jakespertus@berkeley.edu
#SBATCH -a 1-100
python strat_vs_oneaudit_simulations.py
