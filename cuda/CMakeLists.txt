cmake_minimum_required(VERSION 3.20)

enable_language(CUDA)

# CUDA configuration
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# CUDA sources
set(CUDA_SOURCES
    cuda_simple.cu
)

# Create CUDA library
add_library(helix_cuda STATIC ${CUDA_SOURCES})

# Link with CUDA libraries
find_package(CUDAToolkit REQUIRED)
target_link_libraries(helix_cuda 
    PRIVATE 
    CUDA::cudart 
    CUDA::cublas 
    CUDA::curand
)

# Include directories
target_include_directories(helix_cuda 
    PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/cuda
)

# Compile features
# target_compile_features(helix_cuda PUBLIC cuda_std_20)

# Set CUDA architecture
set_property(TARGET helix_cuda PROPERTY CUDA_ARCHITECTURES 75 80 86 89)

# Link with main helix library
target_link_libraries(helix_cuda PRIVATE helix)
