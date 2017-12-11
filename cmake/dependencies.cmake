
set(DEPENDENCY_LIBS "")

# find glog
find_package(Glog)
if (GLOG_FOUND)
  include_directories(${GLOG_INCLUDE_DIR})
  list(APPEND DEPENDENCY_LIBS ${GLOG_LIBRARY})
else()
  message(FATAL_ERROR "glog is not found.")
endif()

#find gflags
find_package(Gflags)
if (GFLAGS_FOUND)
  include_directories(${GFLAGS_INCLUDE_DIR})
  list(APPEND DEPENDENCY_LIBS ${GFLAGS_LIBRARY})
else()
  message(FATAL_ERROR "gflags is not found.")
endif()

# find gtest
find_package(Gtest)
if (GTEST_FOUND)
  include_directories(${GTEST_INCLUDE_DIR})
  list(APPEND DEPENDENCY_LIBS ${GTEST_LIBRARY})
else()
  message(FATAL_ERROR "gtest is not found.")
endif()

# find blas
set(CBLAS_FOUND OFF)
if (NOT CBLAS_FOUND)
  find_package(MKL)
  if (MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIR})
    list(APPEND DEPENDENCY_LIBS ${MKL_LIBRARYS})
    add_definitions(-DPADDLE_USE_MKL)
    set(CBLAS_FOUND ON)
  endif()
endif()

if (NOT CBLAS_FOUND)
  find_package(OpenBLAS)
  if (OPENBLAS_FOUND)
    include_directories(${OPENBLAS_INCLUDE_DIR})
    list(APPEND DEPENDENCY_LIBS ${OPENBLAS_LIBRARY})
    set(CBLAS_FOUND ON)
  endif()
endif()

if (NOT CBLAS_FOUND)
  message(FATAL_ERROR "blas is not found.")
endif()

# find Eigen
find_package(Eigen)
if (EIGEN_FOUND)
  include_directories(${EIGEN_INCLUDE_DIR})
else()
  message(FATAL_ERROR "eigen is not found.")
endif()

# find nnpack
  find_package(NNPACK)
  if (NNPACK_FOUND)
  set(USE_NNPACK ON)
    include_directories(${NNPACK_INCLUDE_DIR})
    list(APPEND DEPENDENCY_LIBS ${NNPACK_LIBRARYS})
  else()
    set(USE_NNPACK OFF)
endif()

# find benchmark
find_package(Benchmark)
if (BENCHMARK_FOUND)
  include_directories(${BENCHMARK_INCLUDE_DIR})
  list(APPEND DEPENDENCY_LIBS ${BENCHMARK_LIBRARY})
else()
  message(FATAL_ERROR "benchmark is not found.")
endif()