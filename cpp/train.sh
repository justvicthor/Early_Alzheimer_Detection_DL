#!/bin/bash -l
#SBATCH -A p200895
#SBATCH -p gpu
#SBATCH -q dev
#SBATCH -J adni_train
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=64
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

# ============ 1. Conda Env =====
WDIR=/project/home/p200895/vitto
source "$WDIR/../conda_base_path/miniconda3/etc/profile.d/conda.sh"
conda activate /project/home/p200895/conda_base_path/miniconda3/envs/trainEnv

# ============ 2. Modules =========
#module load gcc
module avail pytorch libtorch

module load help2man/1.49.3-GCCcore-13.3.0
module load env/release/2024.1
module load env/staging/2024.1
module load PyTorch/2.3.0-foss-2024a-CUDA-12.6.0 
module load GCC/13.3.0

#module load cmake
module load CMake/3.29.3-GCCcore-13.3.0

#module load cuda
module load CUDA/12.6.0

#module load zlib
module load zlib/1.3.1

#module load python
module load Python/3.12.3-GCCcore-13.3.0

# ============ 3. Variables ==========
PROJECT_DIR=/project/home/p200895/alby_C++
export NIFTI_DIR=$PROJECT_DIR/libraries/nifti_clib-3.0.0
#export TORCH_HOME=$PROJECT_DIR/libraries/libtorch
export LD_LIBRARY_PATH=$NIFTI_DIR/build/lib:$LD_LIBRARY_PATH

# ============ 4. Build NIfTI (if necessary) ==========
if [ ! -f "$NIFTI_DIR/build/lib/libniftiio.a" ]; then
  echo "⏳ Compiling NIfTI..."
  # butta via qualsiasi build precedente per evitare cache incoerenti
  rm -rf "$NIFTI_DIR/build"
  mkdir -p "$NIFTI_DIR/build"
  cd "$NIFTI_DIR/build"

  cmake .. \
    || { echo "❌ NIfTI CMake failed"; exit 1; }

  make -j$(nproc) \
    || { echo "❌ NIfTI make failed"; exit 1; }

  echo "✅ NIfTI compiled!"
else
  echo "✅ NIfTI already compiled."
fi


# ============ 5. Project build ========================
cd "$PROJECT_DIR"
rm -rf build && mkdir build && cd build

cmake \
  -DCMAKE_PREFIX_PATH="$TORCH_HOME" \
  -DNIFTI_DIR="$NIFTI_DIR" \
  -D_GLIBCXX_USE_CXX11_ABI=0 \
  .. || { echo "❌ CMake failed"; exit 1; }

make -j$(nproc) || { echo "❌ make failed"; exit 1; }


# ============ 6. Exec =========
echo "🚀 Launching the model..."
cd "$PROJECT_DIR"           
./build/train             
