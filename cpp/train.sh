#!/bin/bash -l
#SBATCH -A p200895
#SBATCH -p gpu
#SBATCH -q dev
#SBATCH -J adni_train_cpp
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

# ============================================================================
#  1. ENVIRONMENT SETUP
# ============================================================================
echo "1. Setting up environment..."
PROJECT_DIR="/path/to/libraries"
PROJECT_DIR_MAIN="/path/to/working_dir"

echo "Loading required modules..."
module purge
module load GCC/13.3.0
module load CMake/3.29.3-GCCcore-13.3.0
module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0
module load zlib/1.3.1
module load help2man/1.49.3-GCCcore-13.3.0
echo "Modules loaded."

# ============================================================================
#  2. VARIABLES AND PATH DEFINITIONS
# ============================================================================
echo "2. Defining project variables..."
export NIFTI_DIR="$PROJECT_DIR/nifti_clib-3.0.0"
export YAMLCPP_DIR="$PROJECT_DIR/yaml-cpp-0.8.0"

# Il passo di installazione creerà una sottodirectory 'install'
export NIFTI_INSTALL_DIR="$NIFTI_DIR/build/install"
export YAMLCPP_INSTALL_DIR="$YAMLCPP_DIR/build/install"

export LD_LIBRARY_PATH="$NIFTI_INSTALL_DIR/lib:$YAMLCPP_INSTALL_DIR/lib:$LD_LIBRARY_PATH"

echo "   - NIFTI_DIR: $NIFTI_DIR"
echo "   - YAMLCPP_DIR: $YAMLCPP_DIR"
echo "   - LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Variables defined."

# ============================================================================
#  3. COMPILING + INSTALLING DEPENDENCIES
# ============================================================================
echo "3. Building and installing dependencies if necessary..."

# --- NIfTI ---
if [ ! -f "$NIFTI_INSTALL_DIR/lib/libniftiio.a" ]; then
  echo "   -> Compiling and installing NIfTI library..."
  rm -rf "$NIFTI_DIR/build"
  mkdir -p "$NIFTI_DIR/build" && cd "$NIFTI_DIR/build"
  cmake .. -DCMAKE_INSTALL_PREFIX="$NIFTI_INSTALL_DIR" -DNIFTI_INSTALL_NO_DOCS=ON || { echo "NIfTI CMake failed"; exit 1; }
  make -j$(nproc) || { echo "NIfTI make failed"; exit 1; }
  make install || { echo "NIfTI make install failed"; exit 1; } 
  echo "   NIfTI compiled and installed successfully."
else
  echo "   NIfTI library already installed."
fi

# --- yaml-cpp ---
if [ ! -f "$YAMLCPP_INSTALL_DIR/lib/libyaml-cpp.a" ]; then
  echo "   -> Compiling and installing yaml-cpp library..."
  rm -rf "$YAMLCPP_DIR/build"
  mkdir -p "$YAMLCPP_DIR/build" && cd "$YAMLCPP_DIR/build"
  cmake .. -DCMAKE_INSTALL_PREFIX="$YAMLCPP_INSTALL_DIR" -DYAML_CPP_BUILD_TESTS=OFF || { echo "yaml-cpp CMake failed"; exit 1; }
  make -j$(nproc) || { echo "yaml-cpp make failed"; exit 1; }
  make install || { echo "yaml-cpp make install failed"; exit 1; } 
  echo "   yaml-cpp compiled and installed successfully."
else
  echo "   yaml-cpp library already installed."
fi
echo "Dependencies are ready."

# ============================================================================
#  4. BUILD
# ============================================================================
echo "4. Building the main project..."
cd "$PROJECT_DIR_MAIN"
rm -rf build && mkdir build && cd build

cmake \
  -DCMAKE_PREFIX_PATH="$YAMLCPP_INSTALL_DIR" \
  -DNIFTI_INSTALL_DIR="$NIFTI_INSTALL_DIR" \
  -DYAMLCPP_INSTALL_DIR="$YAMLCPP_INSTALL_DIR" \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  .. || { echo "Project CMake configuration failed"; exit 1; }

make -j$(nproc) || { echo "Project build failed"; exit 1; }
echo "Project built successfully."

# ============================================================================
#  5. TRAINING EXECUTION
# ============================================================================
echo "5. Launching the training process..."
cd "$PROJECT_DIR_MAIN"

if [ ! -f "./build/train_app" ]; then
    echo "Executable 'train_app' not found in build directory!"
    exit 1
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

./build/train_app ../config.yaml

echo "Training finished!"