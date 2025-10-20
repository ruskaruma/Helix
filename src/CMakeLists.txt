set(HELIX_SOURCES
    common/version.cpp
    core/distance.cpp
    core/memory.cpp
    core/io.cpp
    core/vector_utils.cpp
    core/threading.cpp
    core/kmeans.cpp
    core/manifest.cpp
    index/index_flat.cpp
    index/index_pq.cpp
    index/index_ivf.cpp
    index/index_hnsw.cpp
    quantization/product_quantizer.cpp
    benchmark/dataset.cpp
    benchmark/metrics.cpp
)

add_library(helix SHARED ${HELIX_SOURCES})
target_include_directories(helix PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

install(TARGETS helix
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/helix
    DESTINATION include
)

