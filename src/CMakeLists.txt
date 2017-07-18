

file(GLOB cpp_files . BufferArg.cpp)
list(APPEND cpp_files Function.cpp)
#list(APPEND cpp_files Gemm.cpp)
#list(APPEND cpp_files ConvFunc.cpp)
list(APPEND cpp_files Util.cpp)
if(USE_NNPACK)
  list(APPEND cpp_files NNPACKConvFunc.cpp)
endif()

add_library(function STATIC ${cpp_files})
#add_dependencies(function ${external_project_dependencies})

add_executable(TensorShapeTest TensorShapeTest.cpp)
add_executable(BufferArgTest BufferArgTest.cpp)
add_executable(FunctionTest FunctionTest.cpp)
add_executable(ConvFuncTest ConvFuncTest.cpp)

target_link_libraries(TensorShapeTest ${DEPENDENCY_LIBS})
add_dependencies(TensorShapeTest function)
target_link_libraries(BufferArgTest ${DEPENDENCY_LIBS})
add_dependencies(BufferArgTest function)
target_link_libraries(FunctionTest ${DEPENDENCY_LIBS})
add_dependencies(FunctionTest function)

target_link_libraries(ConvFuncTest
                      "-Wl,--whole-archive"
                      function
                      "-Wl,--no-whole-archive")
target_link_libraries(ConvFuncTest
                      "-Wl,--start-group"
                      ${DEPENDENCY_LIBS} "rt"
                      "-Wl,--end-group")
add_dependencies(ConvFuncTest function)

