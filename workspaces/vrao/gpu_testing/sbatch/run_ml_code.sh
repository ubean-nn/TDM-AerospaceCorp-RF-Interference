#!/bin/bash -l
#SBATCH -N 1         # Number of nodes. ALWAYS set to 1
#SBATCH -n 1         # Number of tasks. ALWAYS set to 1
#SBATCH -c 1         # Number of CPU cores. Can go as high as 128
                     # Each additional CPU core adds around 1.9GB of RAM so
                     # to get more memory, add more CPU cores.
#SBATCH -t 1:0:0     # Number of hours to run
#SBATCH -A cis220051 # The TDM account to charge for this.
#SBATCH -p shared    # Partition to use


# These three lines use the TDM python
module use /anvil/projects/tdm/opt/core
module load tdm
module load python/seminar r/seminar

python3 ml_code.py > alt4_output_model.txt