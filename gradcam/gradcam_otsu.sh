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

echo "INFO: Starting visualization with Grad-CAM for $SCAN_PATH"
WDIR=/path/to/working/dir/
CONDA_BASE_PATH="$WDIR/conda_base_path/miniconda3"
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"

echo "INFO: Activating adni-gradcam environment..."
conda activate adni-gradcam

SCAN_PATH="$WDIR/ADNI_processed/subjects/sub-ADNI021S0753/ses-M006/t1/spm/segmentation/normalized_space/sub-ADNI021S0753_ses-M006_space-Ixi549Space_T1w.nii.gz" #SCAN_PATH_AD2
OUTPUT_DIR="./gradcam_out"

mkdir -p "$OUTPUT_DIR"

echo "INFO: Starting visualization with Grad-CAM for $SCAN_PATH"
python visualize.py \
  --config config.yaml \
  --scan "$SCAN_PATH" \
  --alpha 0.6 \
  --mode otsu \
  --output_dir "$OUTPUT_DIR"

echo "INFO: Job completed."
