cmake_minimum_required(VERSION 3.20)
project(Helix VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(HELIX_BUILD_TESTS "Build unit tests" ON)
option(HELIX_BUILD_BENCH "Build benchmarks" ON)
option(HELIX_BUILD_PYTHON "Build Python bindings" ON)
option(HELIX_BUILD_CUDA "Build CUDA kernels" ON)
option(HELIX_ENABLE_SANITIZERS "Enable address and undefined sanitizers" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

if(HELIX_ENABLE_SANITIZERS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(src)

if(HELIX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HELIX_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(HELIX_BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(HELIX_BUILD_CUDA)
    enable_language(CUDA)
    add_subdirectory(cuda)
endif()

add_subdirectory(examples)

