#!/bin/bash -l
#SBATCH -N 1         # Number of nodes. ALWAYS set to 1
#SBATCH -n 1         # Number of tasks. ALWAYS set to 1
#SBATCH -c 32        # Number of CPU cores. Each GPU gets 32 CPU cores
                     # so we should always ask for 32!
#SBATCH -t 0:15:0     # Run for 1 hour.  Change as needed
#SBATCH -A cis220051-gpu # the TDM account to charge for this
#SBATCH -p gpu-debug       # Must use the "gpu" or "gpu-debug" partition
                     # The wait for "gpu-debug" is shorter, but there is a
                     # maximum runtime of just 15 minutes!
#SBATCH --gpus-per-node=1  # Must use just one GPU.  Do not change!


# These three lines use the TDM python
module use /anvil/projects/tdm/opt/core
module load tdm
module load python/seminar r/seminar

# This is where you specify the program you want to run!
python3 havegpu.py