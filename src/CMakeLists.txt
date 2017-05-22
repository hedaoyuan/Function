

file(GLOB cpp_files . BufferArg.cpp)
list(APPEND cpp_files Function.cpp)
list(APPEND cpp_files Gemm.cpp)
list(APPEND cpp_files ConvFunc.cpp)
list(APPEND cpp_files Util.cpp)

add_library(function STATIC ${cpp_files})
add_dependencies(function ${external_project_dependencies})

add_executable(TensorShapeTest TensorShapeTest.cpp)
add_executable(BufferArgTest BufferArgTest.cpp)
add_executable(FunctionTest FunctionTest.cpp)

set(LIBS ${GLOG_LIBRARIES}
         ${GTEST_LIBRARIES}
         ${GFLAGS_LIBRARIES}
         ${CMAKE_THREAD_LIBS_INIT}
         ${CMAKE_DL_LIBS})

target_link_libraries(TensorShapeTest ${LIBS})
target_link_libraries(BufferArgTest ${LIBS})
target_link_libraries(FunctionTest ${LIBS})
        
