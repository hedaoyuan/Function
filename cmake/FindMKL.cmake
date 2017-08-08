# Find the MKL library
#  MKL_ROOT - where to find MKL include and library.
#
#  MKL_FOUND
#  MKL_INCLUDE_DIR
#  MKL_LIBRARYS

set(INTEL_ROOT "/opt/intel" CACHE PATH "Folder contains intel libs")
set(MKL_ROOT ${INTEL_ROOT}/mkl CACHE PATH "Folder contains MKL")

find_path(MKL_INCLUDE_DIR mkl.h PATHS ${MKL_ROOT}/include)
find_library(MKL_CORE_LIB NAMES mkl_core PATHS
  ${MKL_ROOT}/lib
  ${MKL_ROOT}/lib/intel64)
find_library(MKL_SEQUENTIAL_LIB NAMES mkl_sequential PATHS
  ${MKL_ROOT}/lib
  ${MKL_ROOT}/lib/intel64)
find_library(MKL_INTEL_LP64 NAMES mkl_intel_lp64 PATHS
  ${MKL_ROOT}/lib
  ${MKL_ROOT}/lib/intel64)

set(MKL_LIBRARYS -Wl,--start-group ${MKL_INTEL_LP64} ${MKL_SEQUENTIAL_LIB} ${MKL_CORE_LIB} -Wl,--end-group)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_INCLUDE_DIR MKL_CORE_LIB MKL_SEQUENTIAL_LIB MKL_INTEL_LP64)

mark_as_advanced(MKL_INCLUDE_DIR MKL_CORE_LIB MKL_SEQUENTIAL_LIB MKL_INTEL_LP64)
