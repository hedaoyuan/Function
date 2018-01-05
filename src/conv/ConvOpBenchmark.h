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
#include "Stat.h"

namespace paddle {

void BM_Convolution(benchmark::State& state,
                    const std::string& conv,
                    std::string algo = "auto") {
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
                                .set("algo", algo));

  TensorShape shape0{batchSize, inputChannels, inputSize, inputSize};
  TensorShape shape1{outputChannels, inputChannels, filterSize, filterSize};
  TensorShape shape2{batchSize, outputChannels, outputSize, outputSize};
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape0));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape1));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, shape2));
  test.run(state);
  globalStat.printAllStatus();
  globalStat.reset();
}

/**
 * Args:
 *  input_channels, output_channels, input_size, filter_size, stride, padding
 */
#define CONVOLUTION_BENCHMARK_I(function, algo...)                      \
  BENCHMARK_CAPTURE(BM_Convolution, function#algo, #function, #algo)    \
      ->Args({3, 32, 192, 3, 2, 1})                                     \
      ->Args({3, 32, 224, 3, 2, 1})                                     \
      ->Args({3, 32, 300, 3, 2, 1})                                     \
      ->Args({3, 64, 108, 3, 2, 1})                                     \
      ->Unit(benchmark::kMicrosecond);

#define CONVOLUTION_BENCHMARK_R(function, algo...)                      \
  BENCHMARK_CAPTURE(BM_Convolution, function#algo, #function, #algo)    \
      ->Args({64, 64, 54, 3, 1, 1})                                     \
      ->Args({64, 128, 54, 3, 1, 1})                                    \
      ->Args({128, 128, 27, 3, 1, 1})                                   \
      ->Args({128, 256, 27, 3, 1, 1})                                   \
      ->Args({256, 256, 14, 3, 1, 1})                                   \
      ->Args({256, 512, 14, 3, 1, 1})                                   \
      ->Args({512, 512, 7, 3, 1, 1})                                    \
      ->Unit(benchmark::kMicrosecond);

#define CONVOLUTION_BENCHMARK_P(function, algo...)                      \
  BENCHMARK_CAPTURE(BM_Convolution, function#algo, #function, #algo)    \
      ->Args({64, 128, 54, 1, 1, 0})                                    \
      ->Args({128, 256, 27, 1, 1, 0})                                   \
      ->Args({256, 512, 14, 1, 1, 0})                                   \
      ->Unit(benchmark::kMicrosecond);

#define CONVOLUTION_BENCHMARK_D(function, algo...)                      \
  BENCHMARK_CAPTURE(BM_Convolution, function#algo, #function, #algo)    \
      ->Args({3, 32, 1500, 7, 4, 3})                                    \
      ->Args({16, 32, 375, 3, 1, 1})                                    \
      ->Args({16, 32, 188, 3, 1, 1})                                    \
      ->Unit(benchmark::kMicrosecond);

}  // namespace paddle
