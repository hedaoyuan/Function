/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include "FunctionTest.h"

namespace paddle {

void MatMulFuncBenchmark(const std::string& conv) {
  for (size_t size = 32; size <= 1024; size += 32) {
    LOG(INFO) << "Matrix size is: " << size;
    CpuFunctionBenchmark test(conv,
                              FuncConfig()
                                  .set("aTrans", false)
                                  .set("bTrans", false));
    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{size, size}));
    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{size, size}));
    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{size, size}));
    test.run();
  }
}

TEST(MatMul, Blas) {
  MatMulFuncBenchmark("BlasMatMul-CPU");
}

TEST(MatMul, Eigen) {
  MatMulFuncBenchmark("EigenMatMul-CPU");
}

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  return RUN_ALL_TESTS();
}
