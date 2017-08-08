
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

# find nnpack
if(USE_NNPACK)
  find_package(NNPACK)
  if (NNPACK_FOUND)
    include_directories(${NNPACK_INCLUDE_DIR})
    list(APPEND DEPENDENCY_LIBS ${NNPACK_LIBRARYS})
  else()
    set(USE_NNPACK OFF)
    message(WARNING "NNPACK is not found.")
  endif()
endif()