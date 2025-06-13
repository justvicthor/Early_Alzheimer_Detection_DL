# Requisiti C++ (ADNI Alzheimer Model)

## Librerie necessarie

- **LibTorch** (PyTorch C++ API)
  - [https://pytorch.org/get-started/locally/#libtorch](https://pytorch.org/get-started/locally/#libtorch)
  - Scarica la versione compatibile con C++17 e CUDA 12.6 (se disponibile)

- **NIfTI C Library**
  - Codice: [https://github.com/NIFTI-Imaging/nifti_clib](https://github.com/NIFTI-Imaging/nifti_clib)
  - Oppure su Ubuntu:  
    ```bash
    sudo apt-get install libnifti1-dev
    ```

- **Zlib**
  - Sito ufficiale: [https://zlib.net](https://zlib.net)
  - Oppure:
    ```bash
    sudo apt-get install zlib1g-dev
    ```

## Toolchain richiesta

- **CMake ≥ 3.15**  
- **GCC ≥ 9** (supporto C++17/20)  
- Linux/macOS: `build-essential`  
- Windows: Visual Studio con MSVC

## Build

```bash
mkdir build && cd build
cmake .. -DNIFTI_DIR=/path/to/nifti_clib -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j
