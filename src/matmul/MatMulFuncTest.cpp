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
#include "Function.h"
#include "FunctionTest.h"

namespace paddle {

void TestMatMulFunc(const std::string& conv1,
                    const std::string& conv2) {
  for (const auto transa : {false, true}) {
    for (const auto transb : {false, true}) {
      for (const auto dimM : {1, 10, 45, 100}) {
        for (const auto dimN : {1, 5, 20, 80, 111}) {
          for (const auto dimK : {1, 3, 21, 67, 105}) {
            if (transa && transb) {
              continue;
            }
            LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
                    << " transa=" << transa << " transb=" << transb
                    << " dimM=" << std::setw(5) << dimM
                    << " dimN=" << std::setw(5) << dimN
                    << " dimK=" << std::setw(5) << dimK;
            size_t heightA = (transa == false) ? dimM : dimK;
            size_t widthA = (transa == false) ? dimK : dimM;
            size_t heightB = (transb == false) ? dimK : dimN;
            size_t widthB = (transb == false) ? dimN : dimK;
            size_t heightC = dimM;
            size_t widthC = dimN;

            Compare2Function<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU>
                test(conv1,
                     conv2,
                     FuncConfig()
                         .set("aTrans", transa)
                         .set("bTrans", transb));
            test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{heightA, widthA}));
            test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{heightB, widthB}));
            test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{heightC, widthC}));
            test.run();
          }
        }
      }
    }
  }
}

TEST(MatMul, float) {
  TestMatMulFunc("BlasMatMul-CPU", "EigenMatMul-CPU");
}

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  return RUN_ALL_TESTS();
}
