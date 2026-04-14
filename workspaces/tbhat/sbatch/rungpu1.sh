#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 1:0:0
#SBATCH -A cis220051-gpu
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --output=debug_%j.out

module use /anvil/projects/tdm/opt/core
module load tdm
module load python/seminar r/seminar

# Run unbuffered (-u) so we see logs immediately
python3 -u weeper.py