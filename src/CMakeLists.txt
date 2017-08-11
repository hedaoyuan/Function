
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB cpp_files . Util.cpp)
list(APPEND cpp_files BufferArg.cpp)
list(APPEND cpp_files Function.cpp)
list(APPEND cpp_files Gemm.cpp)
list(APPEND cpp_files conv/ConvFunc.cpp)
list(APPEND cpp_files matmul/BlasMatMulFunc.cpp)
list(APPEND cpp_files matmul/EigenMatMulFunc.cpp)

if(USE_NNPACK)
  list(APPEND cpp_files conv/NNPACKConvFunc.cpp)
endif()

# function library
add_library(function STATIC ${cpp_files})

# test
set(test_files)
list(APPEND test_files TensorShapeTest.cpp)
list(APPEND test_files BufferArgTest.cpp)
list(APPEND test_files FunctionTest.cpp)
list(APPEND test_files conv/ConvFuncTest.cpp)
list(APPEND test_files matmul/MatMulFuncTest.cpp)
list(APPEND test_files matmul/MatMulBenchmark.cpp)

foreach(test_file ${test_files})
  get_filename_component(test ${test_file} NAME_WE)
  add_executable(${test} ${test_file})
  target_link_libraries(${test} "-Wl,--whole-archive" function "-Wl,--no-whole-archive"
                  ${DEPENDENCY_LIBS})
endforeach()
