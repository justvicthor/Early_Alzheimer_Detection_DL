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
WDIR=/path/to/wirking/dir
source "$WDIR/../conda_base_path/miniconda3/etc/profile.d/conda.sh"
conda activate $WDIR/conda_base_path/miniconda3/envs/trainEnv

# ============ 2. Training =========
python train.py --config ../config.yaml

