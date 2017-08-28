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

namespace paddle {

void BM_Convolution(benchmark::State& state, const std::string& conv) {
  size_t inputChannels = state.range(0);
  size_t outputChannels = state.range(1);
  size_t inputSize = state.range(2);
  size_t filterSize = state.range(3);
  size_t stride = state.range(4);
  size_t padding = state.range(5);
  size_t groups = state.range(6);
  size_t batchSize = 1;
  size_t outputSize = (inputSize - filterSize + 2 * padding + stride) / stride;

  std::vector<size_t> paddings = {padding, padding};
  std::vector<size_t> strides = {stride, stride};
  CpuFunctionBenchmark test(conv,
                            FuncConfig()
                                .set("paddings", paddings)
                                .set("strides", strides)
                                .set("groups", groups)
                                .set("algo", (std::string)"auto"));

  TensorShape input{batchSize, inputChannels, inputSize, inputSize};
  TensorShape filter{groups,
                     outputChannels / groups,
                     inputChannels / groups,
                     filterSize,
                     filterSize};
  TensorShape output{batchSize, outputChannels, outputSize, outputSize};
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, input));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, filter));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, output));
  test.run(state);
}

#define DEPTHWISE_CONVOLUTION(function)                         \
  BENCHMARK_CAPTURE(BM_Convolution, function, #function)        \
      ->Args({32, 32, 96, 3, 1, 1, 32})                         \
      ->Args({64, 64, 96, 3, 2, 1, 64})                         \
      ->Args({128, 128, 48, 3, 1, 1, 128})                      \
      ->Args({128, 128, 48, 3, 2, 1, 128})                      \
      ->Args({256, 256, 24, 3, 1, 1, 256})                      \
      ->Args({256, 256, 24, 3, 2, 1, 256})                      \
      ->Args({512, 512, 12, 3, 1, 1, 512})                      \
      ->Args({512, 512, 12, 3, 1, 1, 512})                      \
      ->Args({512, 512, 12, 3, 1, 1, 512})                      \
      ->Args({512, 512, 12, 3, 1, 1, 512})                      \
      ->Args({512, 512, 12, 3, 1, 1, 512})                      \
      ->Args({512, 512, 12, 3, 2, 1, 512})                      \
      ->Args({1024, 1024, 6, 3, 1, 1, 1024})                    \
      ->Unit(benchmark::kMicrosecond);

DEPTHWISE_CONVOLUTION(NeonDepthwiseConv-CPU);
DEPTHWISE_CONVOLUTION(GemmConv-CPU);

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark::RunSpecifiedBenchmarks();

  return 0;
}
