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
#include "ConvOpTest.h"

namespace paddle {

TEST(GemmConv, Forward1) {
  Convolution<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU>(
      "NaiveConv-CPU", "GemmConv-CPU", forward);
}

TEST(GemmConv, Forward2) {
  Convolution2<DEVICE_TYPE_CPU, DEVICE_TYPE_CPU>(
      "NaiveConv-CPU", "GemmConv-CPU", forward);
}

#ifndef PADDLE_ONLY_CPU
TEST(GemmConv, Forward) {
  Convolution<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConv-CPU", "GemmConv-GPU", forward);
  Convolution2<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU>(
      "GemmConv-CPU", "GemmConv-GPU", forward);
}
#endif

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  return RUN_ALL_TESTS();
}
