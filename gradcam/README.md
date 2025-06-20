# gradcam/

This folder contains scripts and outputs for Grad-CAM visualizations:

- `visualize.py`: Python script to generate Grad-CAM visualizations for model interpretability.
- `gradcam_otsu.sh`: Shell script to run Grad-CAM with Otsu thresholding. Usage: `sbatch gradcam_otsu.sh`
- `gradcam_pct.sh`: Shell script to run Grad-CAM with percentile thresholding. Usage: `sbatch gradcam_pct.sh`
- `gradcam_std.sh`: Shell script to run Grad-CAM with standard deviation thresholding. Usage: `sbatch gradcam_std.sh`
- `gradcam_out/`: Output directory for generated Grad-CAM visualizations and results.

Submit the `.sh` scripts to a SLURM cluster using `sbatch` as shown above.
