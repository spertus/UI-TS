#!/bin/bash
#SBATCH --job-name=point_mass_Simulations
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jakespertus@berkeley.edu
#SBATCH -a 1-100
python bernoulli_simulations.py
