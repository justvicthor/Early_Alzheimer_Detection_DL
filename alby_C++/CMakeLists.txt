cmake_minimum_required(VERSION 3.10)
project(ADNI3DCNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# ========= Torch & zlib ======================================================
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# ========= NIfTI =============================================================
# (1) Se l'utente non ha passato -DNIFTI_DIR=..., prova a leggerlo
#     dall'ambiente in cui è stato lanciato cmake.
if(NOT NIFTI_DIR AND DEFINED ENV{NIFTI_DIR})
  set(NIFTI_DIR $ENV{NIFTI_DIR})
endif()

# (2) Header: sorgente + generated
include_directories(
  ${NIFTI_DIR}/niftilib
  ${NIFTI_DIR}/znzlib          
  ${NIFTI_DIR}/build/include
  ${NIFTI_DIR}/build/znzlib  
)


# (3) Directory dove cercare le .a / .so
set(NIFTI_LIB_DIRS
    ${NIFTI_DIR}/build          # root
    ${NIFTI_DIR}/build/lib      # eventuale 'make install'
    ${NIFTI_DIR}/build/niftilib # <-- dove sta libniftiio.a
    ${NIFTI_DIR}/build/znzlib   # <-- dove sta libznz.a
)


link_directories(${NIFTI_LIB_DIRS})

# (4) Fallisci se mancano
find_library(NIFTIIO niftiio PATHS ${NIFTI_LIB_DIRS} REQUIRED)
find_library(ZNZ     znz     PATHS ${NIFTI_LIB_DIRS} REQUIRED)


# ========= Executable ========================================================
add_executable(train train.cpp)

# ========= Link ==============================================================
find_package(Threads REQUIRED)            

target_link_libraries(train
    "${TORCH_LIBRARIES}"
    ${NIFTIIO}
    ${ZNZ}
    ZLIB::ZLIB
    Threads::Threads                    
)


# ========= Include Torch (API C++ di PyTorch) ================================
target_include_directories(train PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
)
