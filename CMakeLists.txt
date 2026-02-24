# mkdir build && cd build
# cmake ..
# cmake --build ../build

# Reference material:
# https://github.com/prabhuomkar/pytorch-cpp

# Cmake version
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Project info
project(deep-residual-network VERSION 1.0.0 LANGUAGES CXX)
set(EXECUTABLE_NAME deep-residual-network)

# Link files
add_executable(${EXECUTABLE_NAME})
target_sources(${EXECUTABLE_NAME} PRIVATE src/main.cpp
                                          src/cifar10.cpp
                                          src/residual_block.cpp
                                          src/transform.cpp
                                          include/residual_block.h
                                          include/resnet.h
                                          include/cifar10.h
                                          include/transform.h
)

target_include_directories(${EXECUTABLE_NAME} PRIVATE include)

#target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})

# Set torch path
set(CMAKE_PREFIX_PATH /root/CISC_830_programmingHomework/libtorch)
# Find pytorch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

# Find MPI package
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
# Use mpi compiler
SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)

set_target_properties(${EXECUTABLE_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# Dataset fetching
option(DOWNLOAD_DATASETS "Automatically download required datasets at build-time." ON)

if(DOWNLOAD_DATASETS)
    set(DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data CACHE PATH "Dataset download directory")
    file(MAKE_DIRECTORY ${DATA_DIR})

    add_custom_target(cifar10 COMMAND ${CMAKE_COMMAND}
            -D DATA_DIR=${DATA_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fetch_cifar10.cmake)
endif()

# Trigger downloading process
if(DOWNLOAD_DATASETS)
    add_dependencies(${EXECUTABLE_NAME} cifar10)
endif()

# pytorch for windows
if(MSVC)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif(MSVC)
