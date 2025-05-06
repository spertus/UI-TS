#!/bin/bash
#SBATCH --job-name=one_audit_betting_simulations
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jakespertus@berkeley.edu
#SBATCH -a 1-20
python oneaudit_simulations.py
