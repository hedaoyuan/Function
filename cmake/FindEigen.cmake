# Find the eigen library
#  EIGEN_ROOT - where to find eigen include.
#
#  EIGEN_FOUND
#  EIGEN_INCLUDE_DIR

set(EIGEN_ROOT "" CACHE PATH "Folder contains eigen")

find_path(EIGEN_INCLUDE_DIR unsupported/Eigen/CXX11/Tensor PATHS ${EIGEN_ROOT} NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(EIGEN DEFAULT_MSG EIGEN_INCLUDE_DIR)

mark_as_advanced(EIGEN_INCLUDE_DIR)