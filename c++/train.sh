#!/bin/bash -l
#SBATCH -A p200895
#SBATCH -p gpu
#SBATCH -q dev
#SBATCH -J adni_train
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --ntasks=64
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

# ============ 1. Conda Env =====
WDIR=/project/home/p200895/vitto
source "$WDIR/../conda_base_path/miniconda3/etc/profile.d/conda.sh"
conda activate /project/home/p200895/conda_base_path/miniconda3/envs/trainEnv

# ============ 2. Modules =========
#module load gcc
module load env/release/2024.1
module load env/staging/2024.1
module load GCC/13.3.0

#module load cmake
module load CMake/3.29.3-GCCcore-13.3.0

#module load cuda
module load CUDA/12.6.0

#module load zlib
module load zlib/1.3.1

#module load python
module load Python/3.12.3-GCCcore-13.3.0

# ============ 3. Variables ======
export TORCH_HOME=$WDIR/../C++/libraries/libtorch
export NIFTI_DIR=$WDIR/../C++/libraries/nifti_clib-3.0.0

# ============ 4. Build NIfTI (if necessary) ====
if [ ! -f "$NIFTI_DIR/build/lib/libniftiio.a" ]; then
  echo "⏳ Compiling NIfTI..."
  mkdir -p "$NIFTI_DIR/build"
  cd "$NIFTI_DIR/build"
  cmake ..
  make -j$(nproc)
  echo "✅ NIfTI's been successfully compiled!"
else
  echo "✅ NIfTI was already compiled."
fi

# ============ 5. Project build ====
cd $WDIR/../C++
rm -rf build
make clean
mkdir -p build
cd build

echo "🛠️ Compiling the project..."
cmake -DCMAKE_PREFIX_PATH=$TORCH_HOME ..
make -j$(nproc)

# ============ 6. Exec =========
echo "🚀 Launching the model..."
./train
