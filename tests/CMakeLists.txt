include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(TEST_SOURCES
    test_main.cpp
    test_version.cpp
    test_distance.cpp
    test_vector_utils.cpp
    test_threading.cpp
    test_index_flat.cpp
    test_index_pq.cpp
    test_product_quantizer.cpp
    test_dataset.cpp
    test_metrics.cpp
    test_kmeans.cpp
    test_index_ivf.cpp
    test_index_hnsw.cpp
)

add_executable(helix_tests ${TEST_SOURCES})
target_link_libraries(helix_tests PRIVATE helix gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(helix_tests)

