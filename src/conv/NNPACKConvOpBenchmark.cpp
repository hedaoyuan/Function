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

#include "FunctionBenchmark.h"

DEFINE_string(algo,
              "auto",
              "The algorithm (auto, ft8x8, ft16x16, wt8x8, "
              "implicit-gemm, or direct) for computing convolution of NNPACK.");

namespace paddle {

void BM_Convolution(benchmark::State& state, const std::string& conv) {
  size_t inputChannels = state.range(0);
  size_t outputChannels = state.range(1);
  size_t inputSize = state.range(2);
  size_t filterSize = state.range(3);
  size_t stride = state.range(4);
  size_t padding = state.range(5);
  size_t batchSize = 1;
  size_t outputSize = (inputSize - filterSize + 2 * padding + stride) / stride;

  std::vector<size_t> paddings = {padding, padding};
  std::vector<size_t> strides = {stride, stride};
  CpuFunctionBenchmark test(conv,
                            FuncConfig()
                                .set("paddings", paddings)
                                .set("strides", strides)
                                .set("groups", (size_t)1)
                                .set("algo", (std::string)FLAGS_algo));

  TensorShape shape0{batchSize, inputChannels, inputSize, inputSize};
  TensorShape shape1{outputChannels, inputChannels, filterSize, filterSize};
  TensorShape shape2{batchSize, outputChannels, outputSize, outputSize};
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape0));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape1));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, shape2));
  test.run(state);
}

BENCHMARK_CAPTURE(BM_Convolution, NNPACKConv-CPU, "NNPACKConv-CPU")
    ->Args({3, 64, 108, 3, 1, 1})
    ->Args({64, 64, 54, 3, 1, 1})
    ->Args({64, 128, 54, 3, 1, 1})
    ->Args({128, 128, 27, 3, 1, 1})
    ->Args({64, 128, 54, 1, 1, 0})
    ->Args({128, 128, 27, 3, 1, 1})
    ->Args({128, 256, 27, 3, 1, 1})
    ->Args({256, 256, 14, 3, 1, 1})
    ->Args({128, 256, 27, 1, 1, 0})
    ->Args({256, 256, 14, 3, 1, 1})
    ->Args({256, 512, 14, 3, 1, 1})
    ->Args({512, 512, 7, 3, 1, 1})
    ->Args({256, 512, 14, 1, 1, 0})
    ->Args({512, 512, 7, 3, 1, 1})
    ->Unit(benchmark::kMicrosecond);

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}
