# Find the gtest library
#  GTEST_ROOT - where to find gtest include and library.
#
#  GTEST_FOUND
#  GTEST_INCLUDE_DIR
#  GTEST_LIBRARY

set(GTEST_ROOT "" CACHE PATH "Folder contains gtest")

find_path(GTEST_INCLUDE_DIR gtest/gtest.h PATHS ${GTEST_ROOT}/include NO_DEFAULT_PATH)
find_library(GTEST_LIBRARY gtest PATHS ${GTEST_ROOT}/lib NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTEST DEFAULT_MSG GTEST_INCLUDE_DIR GTEST_LIBRARY)

mark_as_advanced(GTEST_INCLUDE_DIR GTEST_LIBRARY)