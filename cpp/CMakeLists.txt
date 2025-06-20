cmake_minimum_required(VERSION 3.18)
project(ADNI3D_CNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Torch + zlib
find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# yaml‑cpp
find_path(YAMLCPP_INCLUDE_DIR yaml-cpp/yaml.h PATHS $ENV{YAMLCPP_INSTALL_DIR}/include)
find_library(YAMLCPP_LIB yaml-cpp PATHS $ENV{YAMLCPP_INSTALL_DIR}/lib)
include_directories(${YAMLCPP_INCLUDE_DIR})

# NIfTI
if(NOT NIFTI_INSTALL_DIR AND DEFINED ENV{NIFTI_INSTALL_DIR})
    set(NIFTI_INSTALL_DIR $ENV{NIFTI_INSTALL_DIR})
endif()
find_path(
     NIFTI_INCLUDE_DIRS
     nifti1_io.h
     PATHS         ${NIFTI_INSTALL_DIR}/include
     PATH_SUFFIXES nifti          # <-- cerca anche in …/include/nifti
     REQUIRED)
find_library(NIFTIIO niftiio PATHS ${NIFTI_INSTALL_DIR}/lib REQUIRED)
find_library(ZNZ     znz     PATHS ${NIFTI_INSTALL_DIR}/lib REQUIRED)
include_directories(${NIFTI_INCLUDE_DIRS})

add_executable(train_app train.cpp dataset.h model.h config.h)
add_executable(test_app  test.cpp  dataset.h model.h config.h)

# --- link + include comuni --------------------------------------------------
foreach(tgt train_app test_app)
    target_link_libraries(${tgt}
        ${TORCH_LIBRARIES}      # libtorch
        ${YAMLCPP_LIB}          # yaml-cpp static/lib
        ${NIFTIIO} ${ZNZ}       # nifti
        ZLIB::ZLIB
        Threads::Threads)
    target_include_directories(${tgt} PRIVATE ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(${tgt} PRIVATE TORCH_ENABLE_LLVM)
endforeach()

target_include_directories(train_app PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(train_app PRIVATE TORCH_ENABLE_LLVM)
