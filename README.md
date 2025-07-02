# Early Alzheimer’s Disease Detection from Structural MRIs Using Deep Learning

## 🧠 Overview

![GIF via GradCam](media/CN+MCI+AD.gif)

This project explores the use of **3D Convolutional Neural Networks (3D CNNs)** for early detection of Alzheimer’s Disease (AD) using structural magnetic resonance imaging (sMRI). Inspired by and building upon the work of [Liu et al. (2022)](https://www.nature.com/articles/s41598-022-20674-x), we aim to not only replicate but **enhance their deep learning architecture** using high-performance computing (HPC) resources, particularly the **MeluXina Supercomputer**.

The model classifies subjects into three categories:  
- **Cognitively Normal (CN)**  
- **Mild Cognitive Impairment (MCI)**  
- **Mild AD Dementia (AD)**  

Our work highlights the value of deep learning in automating and improving the diagnostic process for Alzheimer’s Disease, enabling scalable and efficient MRI-based screening.

---

## ✅ Contributions

- Re-implemented and validated Liu et al.’s 3D CNN model on the ADNI dataset   
- Used the **Clinica software suite** for standardized MRI preprocessing in BIDS format 
- Integrated  **data augmentation** techniques (Gaussian blurring, random cropping)  
- Leveraged **MeluXina HPC** for full-scale GPU-based training and evaluation  
- Achieved promising classification results with improved performance on the original paper

---

## 🏗️ Model Architecture

![Model Architecture Placeholder](media/pipeline_inst_batch.png)

> *Note: Figure shows a placeholder representation of the deep learning architecture.*

The model architecture consists of:
- Multiple 3D convolutional blocks with normalization steps and ReLU activations  
- Fully connected layers for classification  
- Cross-entropy loss optimized with Adam  

---

## 📦 Data Pipeline

### Dataset: [ADNI](http://adni.loni.usc.edu/)

| session_id | participant_id | sex | original_study | diagnosis | ... |
|------------|----------------|-----|----------------|-----------|-----|
| ses-M006   | sub-ADNI052S0671 | F   | ADNI1          | LMCI      | ... |
| ses-M000   | sub-ADNI109S0967 | M   | ADNI1          | CN        | ... |
| ses-M000   | sub-ADNI027S0850 | M   | ADNI1          | AD        | ... |

## ⚙️ Preprocessing

To collect the MRI scans and utilize them correctly to train the model please refer to [INSTALL.md](INSTALL.md)

MRI scans were processed using the [Clinica software suite](https://www.clinica.run/):

1. Convert to **BIDS format**
2. Generate a **template** from the training set
3. Apply **spatial normalization** using the template
4. Apply **intensity normalization** to reduce scanner bias

This pipeline ensures data consistency across training, validation, and testing sets.

---

## 🔁 Data Augmentation

To improve generalization and model robustness, we applied:

- **Gaussian Blurring** 
- **Random Cropping**  

Augmentation is performed **on-the-fly** during training.

---

## 💻 Infrastructure: MeluXina Supercomputer

We worked on the GPU-enabled **MeluXina** system provided by EuroHPC.

### SSH Access

```bash
# ~/.ssh/config
Host meluxina
  Hostname login.lxp.lu
  User <user_id>
  Port 8822
  IdentityFile ~/.ssh/id_ed25519_mel
  IdentitiesOnly yes
  ForwardAgent no
```
To connect simply type on the command line
```bash
ssh meluxina
```

## 🚀 Benefits of MeluXina

* Large amount of GPU hours available

* Support for large batch sizes

* GPU parallelization capabilities

* Extended memory for ~1TB datasets  

---

## 🧪 Neural Network Training

- **Loss Function**: Cross-Entropy  
- **Optimizer**: Adam
- **Normalization**: InstanceNorm / BatchNorm   

Most of the model parameters can be tuned by modifying the [config.yaml](config.yaml) file.

---

## 📈 Results

> **NOTE:** Placeholder section. Insert metrics when available.

Expected outcomes based on Liu et al.:

- **AUC > 89.21%** for AD classification  
- Improved performance over ROI-volume/thickness-based models  
- Demonstrated progression prediction capabilities  

---

## 🖼️ Visualizations

We implemented **Grad-CAM** to interpret model decisions and highlight the most discriminative brain regions contributing to each prediction (**CN / MCI / AD**). The visualization pipeline generates **GIF overlays for axial, coronal, and sagittal slices**, color-coded by predicted diagnosis:

- 🟩 **Green** = CN  
- 🟨 **Yellow** = MCI  
- 🟥 **Red** = AD  

Additionally, an **optional hippocampal crop** (based on the AAL atlas) is used to focus on clinically relevant areas, automatically handled via `nilearn`.

To define which part of the Grad-CAM heatmap is considered "active", the script supports multiple **thresholding strategies**, selectable via `--threshold_mode`:

- `pct`: retains the top-N% activations (configurable via `--pct`, e.g., `--pct 60` for 60%)
- `otsu`: uses Otsu’s adaptive method
- `std`: keeps voxels with activation > mean + *k* × std (`--std_k 0.5`, etc.)

This interpretability module offers visual insight into model behavior, improves clinical trust, and highlights class-specific decision regions.

---

## 📂 Project Structure
```bash
├── python/                  # Python Model
│   ├── model.py               # CNN architecture
│   ├── dataset.py             # Dataset preparation
│   ├── test.py                # Test perfomance script
│   ├── test.sh                # Shell script for test.py
│   ├── train.py               # Main training script 
│   └── train.sh               # Shell script for train.py
├── cpp/                     # C++ Model
│   ├── CMakeLists.txt         # CMake config file
│   ├── model.h                # CNN architecture
│   ├── dataset.h              # Dataset preparation
│   ├── config.h               # Parameters config
│   ├── test.cpp               # Test perfomance script
│   ├── test.sh                # Shell script for test.cpp
│   ├── train.cpp              # Main training script
│   └── train.sh               # Shell script for train.cpp
├── utils/                   # Other code
│   ├── plot_metrics.py        # Plot loss, Plot accuracy
│   ├── plot_metrics.sh        # Shell script for plot_metrics.py
│   └── spm_get_doc.m          # MATLAB script for Nipype troubleshooting
├── preprocess/              # Preprocessing scripts
│   ├── run_convert.sh         # ADNI -> BIDS convertion
│   ├── run_adni_preproc.sh    # T1-volume segmentation on training set
│   └── run_adni_valtest.sh    # T1-volume segmentation on val & test set
├── data/                    # Diagnosis datasets
│   ├── participants_Test.tsv  # Subjects in the test set
│   ├── participants_Train.tsv # Subjects in the train set
│   └── participants_Val.tsv   # Subjects in the validation set
├── envs/                    # Conda Environments
│   ├── clinicaEnv.yml         # Conda Env for Clinica
│   ├── gradcamEnv.yml         # Conda Env for Grad-CAM
│   └── trainEnv.yml           # Conda Env for Training
├── gradcam/                 # Grad-CAM visualization
│   ├── gradcam_out/           # Folder containing Grad-CAM outputs
│   ├── visualize.py           # Script for Grad-CAM visualization
│   ├── gradcam_otsu.sh        # Shell script for visualize.py w/Otsu
│   ├── gradcam_std.sh         # Shell script for visualize.py w/std
│   └── gradcam_pct.sh         # Shell script for visualize.py w/pct
├── results/                 # Output files containing Test results
│   ├── cpp/                   # Folder containing C++ results
│   ├── python/                # Folder containing Python results
│   └── Test_Results.ipynb     # Notebook for model evaluation metrics
├── media/                   # Images/GIFs/...
├── config.yaml              # Model hyperparameters
├── INSTALL.md
├── README.md
└── report.pdf
```

## 🙏 Acknowledgements
- **Liu et al.** for their foundational model and research

- **MeluXina Support Team** for infrastructure and consultation

- **Clinica Developers** for powerful neuroimaging tools

- **MOX Lab @ Politecnico di Milano** for support and guidance

## 📬 Contacts
```bash
├── Vittorio Pio Remigio Cozzoli, Student, Politecnico di Milano
│     ├── vittoriopio.cozzoli@mail.polimi.it
├── Tommaso Crippa, Student, Politecnico di Milano
│     ├── tommaso2.crippa@mail.polimi.it
├── Alberto Taddei, Student, Politecnico di Milano
│     ├── alberto4.taddei@mail.polimi.it
```

## License and Ethical Use Notice

All rights reserved.  
This repository contains research code developed for academic purposes only.  
Images and medical data used in this project are derived from publicly available datasets (ADNI), that are not linked to any identifiable individuals.  

Nevertheless, as the subject matter involves sensitive neuroimaging data, we kindly ask users to treat visual outputs with scientific respect.  
Any clinical, diagnostic, or commercial usage of this code or its outputs is strictly prohibited without prior written permission.

If you are a researcher or instructor and wish to reuse this material for non-commercial academic use, please contact the authors.
