#!/bin/bash -l
#SBATCH -A <your_project_account>
#SBATCH -p cpu
#SBATCH -q long
#SBATCH -J adni_preprocess
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH --time=64:00:00
#SBATCH --output=%x_%j.out

# ------------------ 1. Load Required Modules ------------------
module load env/release/<release_version>                   # e.g., 2022.1
module load env/staging/<release_version>
module load MCR/<your_matlab_version>                       # Required by SPM standalone
export MCR_HOME="path/to/MCR"                

# ------------------ 2. Activate Conda with Clinica ------------
CONDA_BASE_DIR="/path/to/miniconda3"
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
conda activate clinicaEnv                 # e.g., clinicaEnv

# ------------------ 3. Configure SPM Standalone ---------------
SPM_STANDALONE_DIR="/path/to/spm_standalone"
export SPMSTANDALONE_HOME="$SPM_STANDALONE_DIR"
export SPM_HOME="$SPMSTANDALONE_HOME/spm12-main"
export SPMMCRCMD="$SPMSTANDALONE_HOME/run_spm12.sh $MCR_HOME script"
export MATLABCMD="$SPMMCRCMD"
export FORCE_SPMMCR=1
export PATH="$SPMSTANDALONE_HOME:$PATH"

# Optional: Remove MATLAB Proxy if present
MATLAB_PROXY_PATH="/apps/USE/easybuild/release/2024.1/software/matlab-proxy/0.23.4-GCCcore-13.3.0/bin"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "^$MATLAB_PROXY_PATH\$" | paste -sd ':' -)

# ------------------ 4. Clean shell stdin / stty ---------------
export TERM=dumb
exec 0</dev/null

# ------------------ 5. Set up Library Paths -------------------
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:.:${MCR_HOME}/runtime/glnxa64"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MCR_HOME}/bin/glnxa64"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MCR_HOME}/sys/os/glnxa64"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MCR_HOME}/sys/opengl/lib/glnxa64"
export LD_LIBRARY_PATH

# ------------------ 6. Run Clinica Preprocessing --------------
BIDS_DIR="/path/to/BIDS_DIRECTORY"
OUT_DIR="/path/to/OUTPUT_DIRECTORY"

# Run preprocessing for validation set
clinica run t1-volume-existing-template \
  "$BIDS_DIR" \
  "$OUT_DIR" \
  TRAIN \
  -tsv /path/to/Val_50.tsv \
  -wd /path/to/WD_val \
  --n_procs="$SLURM_NTASKS"

# Run preprocessing for test set
clinica run t1-volume-existing-template \
  "$BIDS_DIR" \
  "$OUT_DIR" \
  TRAIN \
  -tsv /path/to/Test_50.tsv \
  -wd /path/to/WD_test \
  --n_procs="$SLURM_NTASKS"
