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

#pragma once

#include <memory>
#include <time.h>
#include <string>
#include <cmath>
#include "Function.h"
#include "FunctionTest.h"
#include "MemoryHandle.h"
#include "benchmark/benchmark.h"

namespace paddle {

template <class Allocator>
class FunctionBenchmark {
public:
  FunctionBenchmark(const std::string& name,
                    const FuncConfig& config)
      : name_(name),
        function_(FunctionBase::funcRegistrar_.createByType(name)) {
    function_->init(config);
  }

  ~FunctionBenchmark() {}

  void addInputs(const BufferArg& input) {
    size_t size =
        input.shape().getElements() * sizeOfValuType(input.valueType());
    funcMemory_.emplace_back(std::make_shared<Allocator>(size));
    funcInputs_.emplace_back(std::make_shared<BufferArg>(
        funcMemory_.back()->getBuf(), input.valueType(), input.shape()));
  }

  void addOutputs(const BufferArg& output) {
    size_t size =
        output.shape().getElements() * sizeOfValuType(output.valueType());
    funcMemory_.emplace_back(std::make_shared<Allocator>(size));

    funcOutputs_.emplace_back(
        std::make_shared<BufferArg>(funcMemory_.back()->getBuf(),
                                    output.valueType(),
                                    output.shape(),
                                    ASSIGN_TO));
  }

  void run(benchmark::State& state) {
    // prepare arguments
    Uniform<float> uniform(0.001, 1);
    for (size_t i = 0; i < funcInputs_.size(); i++) {
      uniform(*funcInputs_[i]);
    }

    BufferArgs inArgs;
    BufferArgs outArgs;
    for (auto arg : funcInputs_) {
      inArgs.addArg(*arg);
    }
    for (auto arg : funcOutputs_) {
      outArgs.addArg(*arg);
    }

    function_->calc(inArgs, outArgs);

    double flops = 0.0;
    while (state.KeepRunning()) {
      function_->calc(inArgs, outArgs);
      flops += function_->ops(inArgs, outArgs);
    }
    state.counters["gflops"] = benchmark::Counter(flops, benchmark::Counter::kIsRate);
  }

protected:
  std::string name_;
  std::shared_ptr<FunctionBase> function_;
  std::vector<std::shared_ptr<Allocator>> funcMemory_;
  std::vector<BufferArgPtr> funcInputs_;
  std::vector<BufferArgPtr> funcOutputs_;
};

typedef FunctionBenchmark<CpuMemoryHandle> CpuFunctionBenchmark;

}  // namespace paddle
