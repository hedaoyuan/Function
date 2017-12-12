
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB cpp_files . Util.cpp)
list(APPEND cpp_files BufferArg.cpp)
list(APPEND cpp_files Function.cpp)
list(APPEND cpp_files Gemm.cpp)
list(APPEND cpp_files conv/GemmConvOp.cpp)
list(APPEND cpp_files conv/NaiveConvOp.cpp)
list(APPEND cpp_files conv/Im2ColOp.cpp)
list(APPEND cpp_files conv/NeonDepthwiseConv.cpp)
list(APPEND cpp_files conv/NeonDepthwiseConvTranspose.cpp)
list(APPEND cpp_files matmul/BlasMatMulFunc.cpp)
list(APPEND cpp_files matmul/EigenMatMulFunc.cpp)

if(USE_NNPACK)
  list(APPEND cpp_files conv/NNPACKConvOp.cpp)
endif()

# function library
add_library(function STATIC ${cpp_files})

# test
set(test_files)
list(APPEND test_files TensorShapeTest.cpp)
list(APPEND test_files BufferArgTest.cpp)
list(APPEND test_files FunctionTest.cpp)
list(APPEND test_files conv/GemmConvOpTest.cpp)
list(APPEND test_files conv/GemmConvOpBenchmark.cpp)
list(APPEND test_files conv/DepthwiseConvOpTest.cpp)
list(APPEND test_files conv/DepthwiseConvOpBenchmark.cpp)
list(APPEND test_files matmul/MatMulFuncTest.cpp)
list(APPEND test_files matmul/MatMulBenchmark.cpp)
if(USE_NNPACK)
  list(APPEND test_files conv/NNPACKConvOpTest.cpp)
  list(APPEND test_files conv/NNPACKConvOpBenchmark.cpp)
endif()

foreach(test_file ${test_files})
  get_filename_component(test ${test_file} NAME_WE)
  add_executable(${test} ${test_file})
  target_link_libraries(${test} "-Wl,--whole-archive" function "-Wl,--no-whole-archive"
                  ${DEPENDENCY_LIBS})
endforeach()
