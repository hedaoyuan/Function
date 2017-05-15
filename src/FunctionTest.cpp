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

#include "Function.h"
#include <gtest/gtest.h>

namespace paddle {

/**
 * Some tests case are used to check the consistency between the BufferArg type
 * argument received by Function and the original type argument.
 *
 * Use Case:
 *  TEST() {
 *    Matrix matrix(...);
 *    CheckBufferArg lambda = [=](const BufferArg& arg) {
 *      // check matrix and arg are equivalent
 *      EXPECT_EQ(matrix, arg);
 *    }
 *
 *   BufferArgs argments{matrix...};
 *   std::vector<CheckBufferArg> checkFunc{lambda...};
 *   testBufferArgs(argments, checkFunc);
 *  }
 */
typedef std::function<void(const BufferArg&)> CheckBufferArg;

void testBufferArgs(const BufferArgs& inputs,
                    const std::vector<CheckBufferArg>& check) {
  EXPECT_EQ(inputs.size(), check.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    check[i](inputs[i]);
  }
}

void testBufferArgs(const BufferArgs& inputs, const CheckBufferArg& check) {
  EXPECT_EQ(inputs.size(), 1);
  check(inputs[0]);
}

TEST(Arguments, BufferArg) {
  BufferArg arg(nullptr, VALUE_TYPE_FLOAT, {1, 2, 3});
  CheckBufferArg check = [=](const BufferArg& arg) {
    EXPECT_EQ(arg.shape().ndims(), 3);
    EXPECT_EQ(arg.shape()[0], 1);
    EXPECT_EQ(arg.shape()[1], 2);
    EXPECT_EQ(arg.shape()[2], 3);
  };

  BufferArgs argments;
  argments.addArg(arg);
  testBufferArgs(argments, check);
}

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
