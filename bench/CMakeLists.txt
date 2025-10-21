message(STATUS "Benchmark building enabled")

add_executable(benchmark_flat benchmark_flat.cpp)
target_link_libraries(benchmark_flat PRIVATE helix)

add_executable(benchmark_pq benchmark_pq.cpp)
target_link_libraries(benchmark_pq PRIVATE helix)

# Only build CUDA benchmark if CUDA is enabled
if(HELIX_BUILD_CUDA)
    add_executable(benchmark_cuda_simple benchmark_cuda_simple.cpp)
    target_link_libraries(benchmark_cuda_simple PRIVATE helix helix_cuda)
    target_include_directories(benchmark_cuda_simple PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

