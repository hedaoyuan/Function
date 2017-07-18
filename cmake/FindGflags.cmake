# Find the gflags library
#  GFLAGS_ROOT - where to find glog include and library.
#
#  GFLAGS_FOUND
#  GFLAGS_INCLUDE_DIR
#  GFLAGS_LIBRARY

set(GFLAGS_ROOT "" CACHE PATH "Folder contains gflags")

find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h PATHS ${GFLAGS_ROOT}/include)
find_library(GFLAGS_LIBRARY gflags PATHS ${GFLAGS_ROOT}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GFLAGS DEFAULT_MSG GFLAGS_INCLUDE_DIR GFLAGS_LIBRARY)

mark_as_advanced(GFLAGS_INCLUDE_DIR GFLAGS_LIBRARY)