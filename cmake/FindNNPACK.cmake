# Find the NNPACK library
#  NNPACK_ROOT - where to find NNPACK include and library.
#
#  NNPACK_FOUND
#  NNPACK_INCLUDE_DIR
#  NNPACK_LIBRARYS

set(NNPACK_ROOT "" CACHE PATH "Folder contains NNPACK")

find_path(NNPACK_INCLUDE_DIR nnpack.h PATHS ${NNPACK_ROOT}/include)
find_library(NNPACK_LIB NAMES nnpack PATHS ${NNPACK_ROOT}/lib)
find_library(PTHREADPOOL_LIB NAMES pthreadpool PATHS ${NNPACK_ROOT}/lib)
find_library(NNPACK_UKERNELS_LIB NAMES nnpack_ukernels PATHS ${NNPACK_ROOT}/lib)
find_library(NNPACK_CPUFEATURES_LIB NAMES cpufeatures PATHS ${NNPACK_ROOT}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NNPACK DEFAULT_MSG NNPACK_INCLUDE_DIR NNPACK_LIB PTHREADPOOL_LIB)

if(NNPACK_FOUND)
  set(NNPACK_LIBRARYS)
  list(APPEND NNPACK_LIBRARYS ${NNPACK_LIB} ${PTHREADPOOL_LIB})
  if (NNPACK_UKERNELS_LIB)
    list(APPEND NNPACK_LIBRARYS ${NNPACK_UKERNELS_LIB})
  endif()
  if (NNPACK_CPUFEATURES_LIB)
    list(APPEND NNPACK_LIBRARYS ${NNPACK_CPUFEATURES_LIB})
  endif()
  if(NOT ANDROID)
    list(APPEND NNPACK_LIBRARYS "rt")
  endif()
  mark_as_advanced(NNPACK_INCLUDE_DIR NNPACK_LIB PTHREADPOOL_LIB)
endif()
