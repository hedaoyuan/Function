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

void BM_MatMul(benchmark::State& state, const std::string& conv) {
  size_t M = state.range(0);
  size_t N = state.range(1);
  size_t K = state.range(2);
  CpuFunctionBenchmark test(conv,
                            FuncConfig()
                                .set("aTrans", false)
                                .set("bTrans", false));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{M, K}));
  test.addInputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{K, N}));
  test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, TensorShape{M, N}));
  test.run(state);
}

// Benchmark of Eigen-CPU
BENCHMARK_CAPTURE(BM_MatMul, Eigen-CPU, "EigenMatMul-CPU")
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({96, 96, 96})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({384, 384, 384})
    ->Args({512, 512, 512})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_MatMul, Eigen-CPU, "EigenMatMul-CPU")
    ->Args({64, 9216, 32})
    ->Args({128, 2304, 64})
    ->Args({128, 2304, 128})
    ->Args({256, 576, 128})
    ->Args({256, 576, 256})
    ->Args({512, 144, 256})
    ->Args({512, 144, 512})
    ->Args({1024, 36, 512})
    ->Args({1024, 36, 1024})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_MatMul, Eigen-CPU, "EigenMatMul-CPU")
    ->Args({9, 128, 256})
    ->Args({16, 64, 256})
    ->Args({48, 64, 256})
    ->Args({48, 96, 64})
    ->Args({48, 104, 64})
    ->Args({64, 96, 64})
    ->Args({64, 104, 64})
    ->Args({128, 128, 256})
    ->Unit(benchmark::kMicrosecond);

// Benchmark of Blas
BENCHMARK_CAPTURE(BM_MatMul, Blas-CPU, "BlasMatMul-CPU")
    ->Args({32, 32, 32})
    ->Args({64, 64, 64})
    ->Args({96, 96, 96})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({384, 384, 384})
    ->Args({512, 512, 512})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_MatMul, Blas-CPU, "BlasMatMul-CPU")
    ->Args({64, 9216, 32})
    ->Args({128, 2304, 64})
    ->Args({128, 2304, 128})
    ->Args({256, 576, 128})
    ->Args({256, 576, 256})
    ->Args({512, 144, 256})
    ->Args({512, 144, 512})
    ->Args({1024, 36, 512})
    ->Args({1024, 36, 1024})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(BM_MatMul, Blas-CPU, "BlasMatMul-CPU")
    ->Args({9, 128, 256})
    ->Args({16, 64, 256})
    ->Args({48, 64, 256})
    ->Args({48, 96, 64})
    ->Args({48, 104, 64})
    ->Args({64, 96, 64})
    ->Args({64, 104, 64})
    ->Args({128, 128, 256})
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
