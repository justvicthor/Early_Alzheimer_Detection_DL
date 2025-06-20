# envs/

This folder contains YAML files specifying Python environments for different parts of the project:

- `clinicaEnv.yml`: Environment for running Clinica-related tools or pipelines.
- `gradcamEnv.yml`: Environment for running Grad-CAM visualizations.
- `trainEnv.yml`: Environment for model training and related scripts.

You can create the environments using:

```bash
conda env create -f <env_file>.yml
```
