# preprocess/

This folder contains shell scripts for data preprocessing using the Clinica pipeline:


- `run_convert.sh`: Converts data from ADNI to BIDS. Usage: `sbatch run_convert.sh`
- `run_adni_preprocess.sh`: Does T1-volume segmentation on training set. Usage: `sbatch run_adni_preprocess.sh`
- `run_adni_preprocess_val_test.sh`: Does T1-volume segmentation on validation and test sets. Usage: `sbatch run_adni_preprocess_val_test.sh`

Submit these scripts to a SLURM cluster using `sbatch`.
