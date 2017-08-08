# Find the glog library
#  GLOG_ROOT - where to find glog include and library.
#
#  GLOG_FOUND
#  GLOG_INCLUDE_DIR
#  GLOG_LIBRARY

set(GLOG_ROOT "" CACHE PATH "Folder contains glog")

find_path(GLOG_INCLUDE_DIR glog/logging.h PATHS ${GLOG_ROOT}/include NO_DEFAULT_PATH)
find_library(GLOG_LIBRARY glog PATHS ${GLOG_ROOT}/lib NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLOG DEFAULT_MSG GLOG_INCLUDE_DIR GLOG_LIBRARY)

mark_as_advanced(GLOG_INCLUDE_DIR GLOG_LIBRARY)