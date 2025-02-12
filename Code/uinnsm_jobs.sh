#!/bin/bash
#SBATCH --job-name=UINNSM_Simulations
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jakespertus@berkeley.edu
python error_free_comparison_audits.py
