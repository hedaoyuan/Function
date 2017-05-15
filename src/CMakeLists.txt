set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

cmake_minimum_required(VERSION 3.0)

project(function CXX C)
find_package(Threads REQUIRED)

set(THIRD_PARTY_PATH "${PROJ_ROOT}/third_party" CACHE STRING "A path setting third party libraries download & build directories.")

include(external/gflags)
include(external/glog)
include(external/gtest)

include(flags)
include(util)

add_subdirectory(src)
