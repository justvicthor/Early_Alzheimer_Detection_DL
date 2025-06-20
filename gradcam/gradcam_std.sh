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
WDIR=/project/home/p200895/
CONDA_BASE_PATH="$WDIR/conda_base_path/miniconda3"
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"

echo "INFO: Activating adni-gradcam environment..."
conda activate adni-gradcam

#SCAN_PATH_MCI="/project/home/p200895/ADNI_processed/subjects/sub-ADNI002S0295/ses-M000/t1/spm/segmentation/normalized_space/sub-ADNI002S0295_ses-M000_space-Ixi549Space_T1w.nii.gz" #SCAN_PATH_MCI
#SCAN_PATH="/project/home/p200895/ADNI_processed/subjects/sub-ADNI002S0413/ses-M000/t1/spm/segmentation/normalized_space/sub-ADNI002S0413_ses-M000_space-Ixi549Space_T1w.nii.gz" #SCAN_PATH_CN
#SCAN_PATH_AD="/project/home/p200895/ADNI_processed/subjects/sub-ADNI130S0956/ses-M006/t1/spm/segmentation/normalized_space/sub-ADNI130S0956_ses-M006_space-Ixi549Space_T1w.nii.gz" #SCAN_PATH_AD1
#SCAN_PATH="/project/home/p200895/ADNI_processed/subjects/sub-ADNI024S1307/ses-M000/t1/spm/segmentation/normalized_space/sub-ADNI024S1307_ses-M000_space-Ixi549Space_T1w.nii.gz" #SCAN_PATH_MCI2
SCAN_PATH="/project/home/p200895/ADNI_processed/subjects/sub-ADNI021S0753/ses-M006/t1/spm/segmentation/normalized_space/sub-ADNI021S0753_ses-M006_space-Ixi549Space_T1w.nii.gz" #SCAN_PATH_AD2
OUTPUT_DIR="./gradcam_out"

mkdir -p "$OUTPUT_DIR"

echo "INFO: Starting visualization with Grad-CAM for $SCAN_PATH"
python visualize.py \
  --config config.yaml \
  --scan "$SCAN_PATH" \
  --alpha 0.6 \
  --mode std \
  --k 0.5 \
  --output_dir "$OUTPUT_DIR"

echo "INFO: Job completed."
