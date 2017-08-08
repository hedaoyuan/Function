

file(GLOB cpp_files . BufferArg.cpp)
list(APPEND cpp_files Function.cpp)
list(APPEND cpp_files Gemm.cpp)
list(APPEND cpp_files ConvFunc.cpp)
list(APPEND cpp_files BlasMatMulFunc.cpp)
list(APPEND cpp_files EigenMatMulFunc.cpp)
list(APPEND cpp_files Util.cpp)
if(USE_NNPACK)
  list(APPEND cpp_files NNPACKConvFunc.cpp)
endif()

# function library
add_library(function STATIC ${cpp_files})

# test
set(test_files)
list(APPEND test_files TensorShapeTest.cpp)
list(APPEND test_files BufferArgTest.cpp)
list(APPEND test_files FunctionTest.cpp)
list(APPEND test_files ConvFuncTest.cpp)
list(APPEND test_files MatMulFuncTest.cpp)
list(APPEND test_files MatMulBenchmark.cpp)

foreach(test_file ${test_files})
  get_filename_component(test ${test_file} NAME_WE)
  add_executable(${test} ${test_file})
  target_link_libraries(${test} "-Wl,--whole-archive" function "-Wl,--no-whole-archive"
                  ${DEPENDENCY_LIBS} "rt")
endforeach()
