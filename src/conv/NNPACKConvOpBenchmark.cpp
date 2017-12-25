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

#include "ConvOpBenchmark.h"

namespace paddle {

CONVOLUTION_BENCHMARK_I(NNPACKConv-CPU, implicit-gemm);
CONVOLUTION_BENCHMARK_R(NNPACKConv-CPU, implicit-gemm);
CONVOLUTION_BENCHMARK_R(NNPACKConv-CPU, wt8x8);
CONVOLUTION_BENCHMARK_P(NNPACKConv-CPU, implicit-gemm);
CONVOLUTION_BENCHMARK_D(NNPACKConv-CPU, implicit-gemm);

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}
