set(HELIX_SOURCES
    common/version.cpp
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

