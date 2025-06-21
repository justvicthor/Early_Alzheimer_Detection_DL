#!/bin/bash -l
# --- SLURM JOB CONFIGURATION ---
#SBATCH -A p200895
#SBATCH -p gpu
#SBATCH -q dev
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=8
#SBATCH --time=00:50:00
#SBATCH -J adni_gradcam
#SBATCH --output=%x_%j.out

echo "INFO: Loading Conda..."
WDIR=/path/to/working/dir/
CONDA_BASE_PATH="$WDIR/conda_base_path/miniconda3"
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"

echo "INFO: Activating adni-gradcam environment..."
conda activate adni-gradcam

SCAN_PATH="$WDIR/ADNI_processed/subjects/sub-ADNI027S0256/ses-M012/t1/spm/segmentation/normalized_space/sub-ADNI027S0256_ses-M012_space-Ixi549Space_T1w.nii.gz" #SCAN_PATH_MCI4

OUTPUT_DIR="./gradcam_out"

mkdir -p "$OUTPUT_DIR"

echo "INFO: Starting visualization with Grad-CAM for $SCAN_PATH"
python visualize.py \
  --config config.yaml \
  --scan "$SCAN_PATH" \
  --alpha 0.6 \
  --mode pct \
  --pct 60 \
  --output_dir "$OUTPUT_DIR"

echo "INFO: Job completed."
