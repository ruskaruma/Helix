cmake_minimum_required(VERSION 3.20)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(helix_py MODULE bindings.cpp)

target_link_libraries(helix_py PRIVATE helix)

target_include_directories(helix_py PRIVATE ${CMAKE_SOURCE_DIR}/include)

