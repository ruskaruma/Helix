message(STATUS "Benchmark building enabled")

add_executable(benchmark_flat benchmark_flat.cpp)
target_link_libraries(benchmark_flat PRIVATE helix)

add_executable(benchmark_pq benchmark_pq.cpp)
target_link_libraries(benchmark_pq PRIVATE helix)

