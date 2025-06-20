#!/bin/bash -l
#SBATCH -A p200895
#SBATCH -p gpu
#SBATCH -q dev
#SBATCH -J adni_test_cpp
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8          # meno CPU: solo data-loading
#SBATCH --time=01:00:00             # il test è molto più rapido
#SBATCH --output=%x_%j.out

# ============================================================================
#  1. ENVIRONMENT SETUP
# ============================================================================
echo "1. Setting up environment…"
PROJECT_DIR="/path/to/working_dir"
PROJECT_DIR_MAIN="/path/to/working_dir/cpp_folder"

echo "Loading required modules…"
module purge
module load GCC/13.3.0
module load CMake/3.29.3-GCCcore-13.3.0
module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0   # libtorch
module load zlib/1.3.1
module load help2man/1.49.3-GCCcore-13.3.0
echo "Modules loaded."

# ============================================================================
#  2. VARIABLES AND PATHS
# ============================================================================
echo "2. Defining project variables…"
export NIFTI_DIR="$PROJECT_DIR/nifti_clib-3.0.0"
export YAMLCPP_DIR="$PROJECT_DIR/yaml-cpp-0.8.0"

export NIFTI_INSTALL_DIR="$NIFTI_DIR/build/install"
export YAMLCPP_INSTALL_DIR="$YAMLCPP_DIR/build/install"

export LD_LIBRARY_PATH="$NIFTI_INSTALL_DIR/lib:$YAMLCPP_INSTALL_DIR/lib:$LD_LIBRARY_PATH"

# ============================================================================
#  3. (OPTIONAL) BUILD THIRD-PARTY LIBS         – skipped if already present
# ============================================================================
echo "3. Checking external libraries…"

if [ ! -f "$NIFTI_INSTALL_DIR/lib/libniftiio.a" ]; then
  echo "   Compiling NIfTI…"
  rm -rf "$NIFTI_DIR/build" && mkdir -p "$NIFTI_DIR/build" && cd "$NIFTI_DIR/build"
  cmake .. -DCMAKE_INSTALL_PREFIX="$NIFTI_INSTALL_DIR" -DNIFTI_INSTALL_NO_DOCS=ON
  make -j$(nproc) && make install
else
  echo "   NIfTI library already installed."
fi

if [ ! -f "$YAMLCPP_INSTALL_DIR/lib/libyaml-cpp.a" ]; then
  echo "   Compiling yaml-cpp…"
  rm -rf "$YAMLCPP_DIR/build" && mkdir -p "$YAMLCPP_DIR/build" && cd "$YAMLCPP_DIR/build"
  cmake .. -DCMAKE_INSTALL_PREFIX="$YAMLCPP_INSTALL_DIR" -DYAML_CPP_BUILD_TESTS=OFF
  make -j$(nproc) && make install
else
  echo "   yaml-cpp library already installed."
fi

# ============================================================================
#  4. BUILD PROJECT (if not already built)
# ============================================================================
echo "4. Building the main project…"
cd "$PROJECT_DIR_MAIN"
if [ ! -d build ] || [ ! -f build/test_app ]; then
  rm -rf build && mkdir build && cd build
  cmake \
    -DCMAKE_PREFIX_PATH="$YAMLCPP_INSTALL_DIR" \
    -DNIFTI_INSTALL_DIR="$NIFTI_INSTALL_DIR" \
    -DYAMLCPP_INSTALL_DIR="$YAMLCPP_INSTALL_DIR" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    ..
  make -j$(nproc) test_app
else
  echo "   Build directory already exists – skipping rebuild."
fi
echo "Project ready."

# ============================================================================
#  5. TEST EXECUTION
# ============================================================================
echo "5. Launching the test process…"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$PROJECT_DIR_MAIN/build"      # siamo già qui
cd ..                             # <-- torni alla root
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build \
    ./build/test_app ./config.yaml