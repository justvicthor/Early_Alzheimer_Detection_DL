#!/bin/bash -l
#SBATCH -A <your_project_account> 
#SBATCH --job-name=adni_conv
#SBATCH -p cpu
#SBATCH -q long
#SBATCH -J adni_preprocess
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=16G
#SBATCH --time=64:00:00
#SBATCH --output=%x_%j.out

module load anaconda3
conda activate clinicaEnv
clinica -v convert adni-to-bids './ADNI' 'Dir/To/Downloaded/Data' './ADNI_converted' -m T1