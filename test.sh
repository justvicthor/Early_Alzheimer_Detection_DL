#!/bin/bash -l
#SBATCH -A p200895
#SBATCH -p gpu
#SBATCH -q dev
#SBATCH -J adni_train
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=64
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out


# ============ 1. Conda Env =====
WDIR=/project/home/p200895/vitto
source "$WDIR/../conda_base_path/miniconda3/etc/profile.d/conda.sh"
conda activate /project/home/p200895/conda_base_path/miniconda3/envs/trainEnv

# ============ 2. Testing =========
python test.py --config config.yaml

