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
#include <malloc.h>

namespace paddle {

class MemoryHandle {
protected:
  explicit MemoryHandle(size_t size) : size_(size), buf_(nullptr) {}
  virtual ~MemoryHandle() {}

public:
  void* getBuf() const { return buf_; }
  size_t getSize() const { return size_; }

protected:
  // address returned by GPU/CPU memory allocation
  size_t size_;
  void* buf_;
};

/**
 * Wrapper class for raw cpu memory handle.
 * The raw handle will be released at destructor
 */
class CpuMemoryHandle : public MemoryHandle {
public:
  explicit CpuMemoryHandle(size_t size) : MemoryHandle(size) {
    buf_ = memalign(32ul, size);
    CHECK(buf_) << "Fail to allocate CPU memory: size=" << size;
    if (size > 1000 * 1000 * 1000) {
      LOG(INFO) << "CpuMemoryHandle alloc big buf size=" << size;
    }
  }
  virtual ~CpuMemoryHandle() {
    if (buf_) {
      free(buf_);
    }
  }
};

typedef std::shared_ptr<CpuMemoryHandle> CpuMemHandlePtr;

typedef std::shared_ptr<BufferArg> BufferArgPtr;

/**
 * \brief A class for compare the two implementations of a Function.
 *
 *
 * Use case:
 *  // Initializes a test object, the corresponding cpu and gpu Function
 *  // are constructed according to FunctionName and FuncConfig.
 *  FunctionCompare test(FunctionName1, FunctionName2, FuncConfig);
 *  // Prepare inputs and outputs arguments.
 *  // Here the input and output can not contain real data,
 *  // only contains the argument type and shape.
 *  test.addInputs(input1);
 *  test.addInputs(input2);
 *  test.addOutputs(output1);
 *  test.addOutputs(output2);
 *  // Run.
 *  // Will according to the type and shape of arguments(inputs_/outputs_),
 *  // automatic initialization cpu and gpu function required arguments
 *  // (cpuInputs_/cpuOutputs_/gpuInputs_/gpuOutputs_).
 *  // Call the CPU and GPU Function calculation results.
 *  // Compares CPU and GPU calculation results for consistency.
 *  test.run();
 */
class FunctionCompare {
public:
  FunctionCompare(const std::string& name1,
                  const std::string& name2,
                  const FuncConfig& config)
      : function1_(FunctionBase::funcRegistrar_.createByType(name1)),
        function2_(FunctionBase::funcRegistrar_.createByType(name2)) {
    function1_->init(config);
    function2_->init(config);
  }

  ~FunctionCompare() {}

  // input need only contains shape, do not contains data.
  void addInputs(const BufferArg& input) {
    size_t size =
        input.shape().getElements() * sizeOfValuType(input.valueType());

    func1Memory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));
    func2Memory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));

    func1Inputs_.emplace_back(std::make_shared<BufferArg>(
        func1Memory_.back()->getBuf(), input.valueType(), input.shape()));
    func2Inputs_.emplace_back(std::make_shared<BufferArg>(
        func2Memory_.back()->getBuf(), input.valueType(), input.shape()));
  }

  // output need only contains shape, do not contains data.
  void addOutputs(const BufferArg& output) {
    size_t size =
        output.shape().getElements() * sizeOfValuType(output.valueType());
    func1Memory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));
    func2Memory_.emplace_back(std::make_shared<CpuMemoryHandle>(size));

    func1Outputs_.emplace_back(
        std::make_shared<BufferArg>(func1Memory_.back()->getBuf(),
                                    output.valueType(),
                                    output.shape(),
                                    ASSIGN_TO));
    func2Outputs_.emplace_back(
        std::make_shared<BufferArg>(func2Memory_.back()->getBuf(),
                                    output.valueType(),
                                    output.shape(),
                                    ASSIGN_TO));
  }

  void run() {
    // prepare cpu/gpu arguments
    initInputs();

    // function calculate
    auto callFunction = [](FunctionBase* function,
                           std::vector<BufferArgPtr>& inputs,
                           std::vector<BufferArgPtr>& outputs) {
      BufferArgs inArgs;
      BufferArgs outArgs;
      for (auto arg : inputs) {
        inArgs.addArg(*arg);
      }
      for (auto arg : outputs) {
        outArgs.addArg(*arg);
      }
      function->calc(inArgs, outArgs);
    };

    callFunction(function1_.get(), func1Inputs_, func1Outputs_);
    callFunction(function2_.get(), func2Inputs_, func2Outputs_);

    // check outputs and inouts
    compareOutputs();
  }

protected:
  void initInputs() {
    for (size_t i = 0; i < func1Inputs_.size(); i++) {
      initArg(*func1Inputs_[i]);
#if 0
      // TODO: Need a BufferCopy used to copy from one BufferArg to another.
      CpuVector cpuVector(func1Inputs_[i]->shape().getElements(),
                          (real*)func1Inputs_[i]->data());
      GpuVector gpuVector(func2Inputs_[i]->shape().getElements(),
                          (real*)func2Inputs_[i]->data());

      gpuVector.copyFrom(cpuVector);
#endif
    }
  }

  void compareOutputs() {
    for (size_t i = 0; i < func1Outputs_.size(); i++) {
#if 0
      // TODO, Need a BufferCheck used to compare the two buffers.
      auto cpu = func1Outputs_[i];
      auto gpu = func2Outputs_[i];
      CpuVector cpuVector(cpu->shape().getElements(), (real*)cpu->data());
      GpuVector gpuVector(cpu->shape().getElements(), (real*)gpu->data());

      autotest::TensorCheckErr(cpuVector, gpuVector);
#endif
    }
  }

  // only init cpu argument, gpu argument copy from cpu argument.
  void initArg(BufferArg& arg) {
#if 0
    CpuVector vector(arg.shape().getElements(), (real*)arg.data());
    vector.uniform(0.001, 1);
#endif
  }

protected:
  std::shared_ptr<FunctionBase> function1_;
  std::shared_ptr<FunctionBase> function2_;
  std::vector<CpuMemHandlePtr> func1Memory_;
  std::vector<CpuMemHandlePtr> func2Memory_;
  std::vector<BufferArgPtr> func1Inputs_;
  std::vector<BufferArgPtr> func1Outputs_;
  std::vector<BufferArgPtr> func2Inputs_;
  std::vector<BufferArgPtr> func2Outputs_;
};

}  // namespace paddle
