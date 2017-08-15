# Find the benchmark library
#  BENCHMARK_ROOT - where to find benchmark include and library.
#
#  BENCHMARK_FOUND
#  BENCHMARK_INCLUDE_DIR
#  BENCHMARK_LIBRARY

set(BENCHMARK_ROOT "" CACHE PATH "Folder contains benchmark")

find_path(BENCHMARK_INCLUDE_DIR benchmark/benchmark.h PATHS ${BENCHMARK_ROOT}/include NO_DEFAULT_PATH)
find_library(BENCHMARK_LIBRARY benchmark PATHS ${BENCHMARK_ROOT}/lib NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BENCHMARK DEFAULT_MSG BENCHMARK_INCLUDE_DIR BENCHMARK_LIBRARY)

mark_as_advanced(BENCHMARK_INCLUDE_DIR BENCHMARK_LIBRARY)