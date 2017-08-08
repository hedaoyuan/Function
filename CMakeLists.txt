cmake_minimum_required(VERSION 3.0)

project(function CXX C)
find_package(Threads REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(THIRD_PARTY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    CACHE STRING "A path setting third party libraries download & build directories.")

option(USE_NNPACK   "Compile function with NNPACK library" OFF)

include(dependencies)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
list(APPEND DEPENDENCY_LIBS ${CMAKE_THREAD_LIBS_INIT})

add_subdirectory(src)
