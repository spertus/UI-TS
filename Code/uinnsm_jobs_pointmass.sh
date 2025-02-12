#!/bin/bash
#SBATCH --job-name=point_mass_Simulations
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jakespertus@berkeley.edu
python point_mass_ui-tsm.py
