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
#include "FunctionBenchmark.h"

namespace paddle {

void BasicBenchmark(const std::string& conv) {
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
  BasicBenchmark("BlasMatMul-CPU");
}

TEST(MatMul, Eigen) {
  BasicBenchmark("EigenMatMul-CPU");
}

void TypicalCase(const std::string& conv, size_t M, size_t N, size_t K) {
  LOG(INFO) << "MxNxK: " << M << "x" << N << "x" << K;
  CpuFunctionBenchmark test(conv,
                            FuncConfig()
                                .set("aTrans", false)
                                .set("bTrans", false));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{M, K}));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{K, N}));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{M, N}));
  test.run();
}

void Case1(const std::string& conv) {
  TypicalCase(conv, 64, 9216, 32);
  TypicalCase(conv, 128, 2304, 64);
  TypicalCase(conv, 128, 2304, 128);
  TypicalCase(conv, 256, 576, 128);
  TypicalCase(conv, 256, 576, 256);
  TypicalCase(conv, 512, 144, 256);
  TypicalCase(conv, 512, 144, 512);
  TypicalCase(conv, 1024, 36, 512);
  TypicalCase(conv, 1024, 36, 1024);
}

TEST(Blas, Case1) {
  Case1("BlasMatMul-CPU");
}

TEST(Eigen, Case1) {
  Case1("EigenMatMul-CPU");
}

void Case2(const std::string& conv) {
  TypicalCase(conv, 9, 128, 256);
  TypicalCase(conv, 16, 64, 256);
  TypicalCase(conv, 48, 64, 256);
  TypicalCase(conv, 48, 96, 64);
  TypicalCase(conv, 48, 104, 64);
  TypicalCase(conv, 64, 96, 64);
  TypicalCase(conv, 64, 104, 64);
  TypicalCase(conv, 128, 128, 256);
}

TEST(Blas, Case2) {
  Case2("BlasMatMul-CPU");
}

TEST(Eigen, Case2) {
  Case2("EigenMatMul-CPU");
}

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  return RUN_ALL_TESTS();
}
