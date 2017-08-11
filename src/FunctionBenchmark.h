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
#include "MemoryHandle.h"
#include <memory>
#include <time.h>
#include <string>
#include <cmath>

namespace paddle {

template <class Allocator>
class FunctionBenchmark {
public:
  FunctionBenchmark(const std::string& name,
                    const FuncConfig& config)
      : name_(name),
        function_(FunctionBase::funcRegistrar_.createByType(name)) {
    function_->init(config);
    cacheSize_ = 32 * 1024 * 1024; // 32M
    memory_ = std::make_shared<Allocator>(cacheSize_ * 4);
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

  void run(int iter = 10) {
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

    {
      struct timespec tp_start, tp_end;
      float  totalms = 0.0;
      for (int i = 0; i < iter; i++) {
        flushCache();

        clock_gettime(CLOCK_MONOTONIC, &tp_start);
        function_->calc(inArgs, outArgs);
        clock_gettime(CLOCK_MONOTONIC, &tp_end);
        totalms += ((tp_end.tv_nsec - tp_start.tv_nsec)/1000000.0f);
        totalms += (tp_end.tv_sec - tp_start.tv_sec)*1000;
      }
      totalms /= iter;
      double gflops = (double)(function_->ops(inArgs, outArgs)) / 1e6 / totalms;
      LOG(INFO) << name_ << " time(ms): " << totalms << " gflops: " << gflops;
    }
  }

  int flushCache() {
    int value = 0;
    int* data = (int*)memory_->getBuf();
    for (size_t i = 0; i < cacheSize_; i += 64) {
      value += data[i];
    }
    return value;
  }

protected:
  std::string name_;
  std::shared_ptr<FunctionBase> function_;
  std::vector<std::shared_ptr<Allocator>> funcMemory_;
  std::vector<BufferArgPtr> funcInputs_;
  std::vector<BufferArgPtr> funcOutputs_;
  std::shared_ptr<Allocator> memory_;
  size_t cacheSize_;
};

typedef FunctionBenchmark<CpuMemoryHandle> CpuFunctionBenchmark;

}  // namespace paddle
