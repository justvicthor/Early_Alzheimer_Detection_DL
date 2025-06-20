#!/bin/bash -l
#SBATCH -A <your_project_account>
#SBATCH -p cpu
#SBATCH -q long
#SBATCH -J adni_preprocess
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH --time=64:00:00
#SBATCH --output=%x_%j.out

# ------------------ 1. Load Environment Modules ------------------
module load env/release/<release_version>                       # e.g., 2022.1
module load env/staging/<release_version>
module load MCR/<your_matlab_version>                           # Required by SPM standalone
export MCR_HOME="path/to/MCR"

# ------------------ 2. Activate Conda/Clinica ------------------
CONDA_BASE_DIR=/path/to/miniconda3
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
conda activate clinicaEnv                         # Your conda env with Clinica installed

# ------------------ 3. Configure SPM Standalone ------------------
export SPMSTANDALONE_HOME="/path/to/spm_standalone"
export SPM_HOME="$SPMSTANDALONE_HOME/spm12-main"
export SPMMCRCMD="$SPMSTANDALONE_HOME/run_spm12.sh $MCR_HOME script"
export MATLABCMD="$SPMMCRCMD"
export FORCE_SPMMCR=1
export PATH="$SPMSTANDALONE_HOME:$PATH"

# Optional: Remove MATLAB Proxy if present
MATLAB_PROXY_PATH="/apps/USE/easybuild/release/2024.1/software/matlab-proxy/0.23.4-GCCcore-13.3.0/bin"
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "^$MATLAB_PROXY_PATH\$" | paste -sd ':' -)

# ------------------ 4. Disable stty + color ------------------
export TERM=dumb
exec 0</dev/null

# ------------------ 5. Fix LD_LIBRARY_PATH for MCR ------------------
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MCR_HOME/runtime/glnxa64"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MCR_HOME/bin/glnxa64"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MCR_HOME/sys/os/glnxa64"
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MCR_HOME/sys/opengl/lib/glnxa64"
export LD_LIBRARY_PATH

# ------------------ 6. Run Clinica ------------------
clinica run t1-volume \
  /path/to/BIDS_DIRECTORY \
  /path/to/ADNI_processed \
  SESSION_NAME \
  -tsv /path/to/participants.tsv \
  -wd /path/to/CLINICA_WORKING_DIR \
  -np "$SLURM_NTASKS"
