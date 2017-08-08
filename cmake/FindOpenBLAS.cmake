# Find the OpenBlas library
#  OPENBLAS_ROOT - where to find OpenBlas include and library.
#
#  OPENBLAS_FOUND
#  OPENBLAS_INCLUDE_DIR
#  OPENBLAS_LIBRARY

set(OPENBLAS_ROOT "" CACHE PATH "Folder contains OpenBlas")

find_path(OPENBLAS_INCLUDE_DIR cblas.h PATHS ${OPENBLAS_ROOT}/include NO_DEFAULT_PATH)
find_library(OPENBLAS_LIBRARY openblas PATHS ${OPENBLAS_ROOT}/lib NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENBLAS DEFAULT_MSG OPENBLAS_INCLUDE_DIR OPENBLAS_LIBRARY)

mark_as_advanced(OPENBLAS_INCLUDE_DIR OPENBLAS_LIBRARY)