## Download ADNI data
To get the same MRI scans we utilized in this repository, you need to access them through the Alzheimer's Disease Neuroimaging Initiative (ADNI).

1. Request approval and register at [ADNI website](https://ida.loni.usc.edu/login.jsp?project=ADNI).

2. To select the dataset utilized in this project,from the main page click on `PROJECTS` and `ADNI`. To get the imaging data, click on `Search & Download` and choose `Data collections`. Then click `Other shared collections`, `ADNI`, and choose `ADNI1:Complete 1Yr 1.5T`, the dataset with 2294 scans. 

3. To finally download the MRI scans click on `Advanced Download`. We advise you to group files as 10 zip files. To download the clinical data, click on `Download` and choose `Study Data`. Select all the csv files which are present in `ALL` by ticking `Select ALL` tabular data and click Download.

## Data Preprocessing
To correctly utilize the MRI scans provided by ADNI, there are a few data preprocessing steps that are needed to be done with the Clinica software.

> *As of June 2025, the ADNI data portal (LONI) is undergoing changes that affect compatibility with Clinica's adni-to-bids converter. The Clinica team recommends using clinical data downloaded before September 2024. Due to ADNI’s data-use restrictions, we cannot include these files in the repository. Feel free to contact us, we can provide the exact set used in our pipeline upon verification.*

1. **Activate Conda Environment**:To correctly utilize the Clinica package, set up the Conda environment using the provided `envs/clinicaEnv.yml` file:
```bash
conda env create -f clinicaEnv.yml
```

2. **Convert data into BIDS format**: please read the docs on [Clinica adni-to-bids converter](https://aramislab.paris.inria.fr/clinica/docs/public/dev/Converters/ADNI2BIDS/), and install required softwares and the required clinical files. This step is done to convert the ADNI files into BIDS, the standard format used in Clinica.
When everything is installed correctly utilize
```
run_convert.sh
```

3. **preprocess converted and splitted data**: for this step install the needed software in the docs of the [Clinica T1-Volume pipeline](https://aramislab.paris.inria.fr/clinica/docs/public/dev/Pipelines/T1_Volume/). This preprocessing step performs tissue segmentation, bias correction, and spatial normalization of T1-weighted MRI images.
After completing this step correctly, run the following scripts:
```
run_adni_preprocess.sh
```
For val and test use:
```
run_adni_preprocess_val_test.sh
```

## Final steps
After these steps it is now possible to train the neural networks on the MRI scans. To install all the libraries needed for the model to work, simply set up the Conda environment using the provided `envs/trainEnv.yml` file:
```bash
conda env create -f trainEnv.yml
```


## Troubleshooting 

* Be careful when installing SPM software to select the SPM-12 version which is compatible with Clinica instead of the latest versions. If MATLAB is not available in your machine, select the *standalone* version with also the MATLAB Runtime Compiler (MCR) to perform the preprocessing.  

* If during the *adni_to_bids* conversion you get the following error
```OSError: This docstring was not generated by Nipype!``` make sure to add the updated version of the Nipype library on MATLAB