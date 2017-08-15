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

#include <gtest/gtest.h>
#include "Function.h"
#include "MemoryHandle.h"
#include <memory>
#include <string>
#include <cmath>

namespace paddle {

template<class T>
class RandNorm {
public:
  RandNorm(T mean, T std) : mean_(mean), std_(std) {}
  void operator()(BufferArg& arg) {
    CHECK(arg.valueType() == DataType<T>::value);
    size_t size = arg.shape().getElements();
    T* data = arg.data<T>();
    // unsigned int* seed = ThreadLocalRand::getSeed();
    // auto rand1 = [&]() { return (1. + ::rand_r(seed)) * (1. / (1. + RAND_MAX)); };
    auto rand1 = [&]() { return (1. + ::rand()) * (1. / (1. + RAND_MAX)); };
    for (size_t i = 0; i < size - 1; i += 2) {
      T r1 = rand1();
      r1 = std::sqrt(-2 * std::log(r1));
      T r2 = rand1();
      data[i] = mean_ + std_ * r1 * cos(2 * M_PI * r2);
      data[i + 1] = mean_ + std_ * r1 * sin(2 * M_PI * r2);
    }
    T r1 = rand1();
    r1 = std::sqrt(-2 * std::log(r1));
    T r2 = rand1();
    data[size - 1] = mean_ + std_ * r1 * cos(2 * M_PI * r2);
  }
private:
  T mean_;
  T std_;
};

template<class T>
class Uniform {
public:
  Uniform(T left, T right) : left_(left), right_(right) {}
  void operator()(BufferArg& arg) {
    CHECK(arg.valueType() == DataType<T>::value);
    size_t size = arg.shape().getElements();
    T* data = arg.data<T>();
    T range = right_ - left_;
    // unsigned int* seed = ThreadLocalRand::getSeed();
    // auto rand1 = [&]() { return ::rand_r(seed) * (1. / (1. + RAND_MAX)); };
    auto rand1 = [&]() { return ::rand() * (1. / (1. + RAND_MAX)); };
    for (size_t i = 0; i < size; ++i) {
      data[i] = rand1() * range + left_;
    }
  }
private:
  T left_;
  T right_;
};

class CopyArgument {
public:
  void operator()(const BufferArg& input, BufferArg& output) {
    CHECK_EQ(input.valueType(), output.valueType());
    CHECK_LE(input.shape().getElements(), output.shape().getElements());

    const void* src = input.data();
    void* dest = output.data();
    memcpy(dest, src,
      input.shape().getElements() * sizeOfValuType(input.valueType()));
  }
};

template <class T>
class AssertEqual {
public:
  AssertEqual(T err = 0) : err_(err) {}

  inline bool operator()(T a, T b) {
    if (err_ == 0) {
      if (a != b) {
        return false;
      }
    } else {
      if (std::fabs(a - b) > err_) {
        if ((std::fabs(a - b) / std::fabs(a)) > (err_ / 10.0f)) {
          return false;
        }
      }
    }

    return true;
  }

private:
  T err_;
};

class CheckArgument {
public:
  CheckArgument() : compare_(1e-4) {}
  void operator()(const BufferArg& arg1, const BufferArg& arg2) {
    CHECK_EQ(arg1.valueType(), arg2.valueType());
    CHECK(arg1.shape() == arg2.shape());

    size_t size = arg1.shape().getElements();
    // TODO:
    const float* data1 = arg1.data<float>();
    const float* data2 = arg2.data<float>();
    int count = 0;
    for (size_t i = 0; i < size; i++) {
      float a = data1[i];
      float b = data2[i];
      if (!compare_(a, b)) {
        count++;
      }
    }
    EXPECT_EQ(count, 0) << "There are " << count << " different element.";
  }
private:
  AssertEqual<float> compare_;
};

namespace test {
template <DeviceType DType>
struct Allocator;

template <>
struct Allocator<DEVICE_TYPE_CPU> {
  using type = CpuMemoryHandle;
};

template <>
struct Allocator<DEVICE_TYPE_GPU> {
  using type = GpuMemoryHandle;
};

}  // namespace test

typedef std::shared_ptr<BufferArg> BufferArgPtr;

/**
 * \brief A class for comparing two Functions of different implementations.
 *        For example, can be used to compare the CPU and GPU implementation
 *        of the function is consistent.
 *
 * Use case:
 *  // Initializes a test object, the corresponding cpu and gpu Function
 *  // are constructed according to FunctionName and FuncConfig.
 *  CpuGpuFuncCompare test(FunctionName, FuncConfig);
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
template <DeviceType DType1, DeviceType DType2>
class Compare2Function {
public:
  typedef typename test::Allocator<DType1>::type Allocator1;
  typedef typename test::Allocator<DType2>::type Allocator2;

  Compare2Function(const std::string& name1,
                   const std::string& name2,
                   const FuncConfig& config)
      : function1_(FunctionBase::funcRegistrar_.createByType(name1)),
        function2_(FunctionBase::funcRegistrar_.createByType(name2)) {
    function1_->init(config);
    function2_->init(config);
  }

  ~Compare2Function() {}

  // input need only contains shape, do not contains data.
  void addInputs(const BufferArg& input) {
    size_t size =
        input.shape().getElements() * sizeOfValuType(input.valueType());

    func1Memory_.emplace_back(std::make_shared<Allocator1>(size));
    func2Memory_.emplace_back(std::make_shared<Allocator2>(size));

    func1Inputs_.emplace_back(std::make_shared<BufferArg>(
        func1Memory_.back()->getBuf(), input.valueType(), input.shape()));
    func2Inputs_.emplace_back(std::make_shared<BufferArg>(
        func2Memory_.back()->getBuf(), input.valueType(), input.shape()));
  }

  // output need only contains shape, do not contains data.
  void addOutputs(const BufferArg& output, ArgType argType = ASSIGN_TO) {
    size_t size =
        output.shape().getElements() * sizeOfValuType(output.valueType());
    func1Memory_.emplace_back(std::make_shared<Allocator1>(size));
    func2Memory_.emplace_back(std::make_shared<Allocator2>(size));

    func1Outputs_.emplace_back(
        std::make_shared<BufferArg>(func1Memory_.back()->getBuf(),
                                    output.valueType(),
                                    output.shape(),
                                    argType));
    func2Outputs_.emplace_back(
        std::make_shared<BufferArg>(func2Memory_.back()->getBuf(),
                                    output.valueType(),
                                    output.shape(),
                                    argType));
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

    // check outputs
    compareOutputs();
  }

  std::shared_ptr<FunctionBase> getFunction1() const { return function1_; }

  std::shared_ptr<FunctionBase> getFunction2() const { return function2_; }

protected:
  // only init cpu argument, gpu argument copy from cpu argument.
  void initInputs() {
    // TODO: Replace it with a support for more types of Uniform functor.
    Uniform<float> uniform(0.001, 1);
    CopyArgument copy;
    for (size_t i = 0; i < func1Inputs_.size(); i++) {
      uniform(*func1Inputs_[i]);
      copy(*func1Inputs_[i], *func2Inputs_[i]);
    }
  }

  void compareOutputs() {
    CheckArgument check;
    for (size_t i = 0; i < func1Outputs_.size(); i++) {
      check(*func1Outputs_[i], *func2Outputs_[i]);
    }
  }

protected:
  std::shared_ptr<FunctionBase> function1_;
  std::shared_ptr<FunctionBase> function2_;
  std::vector<std::shared_ptr<Allocator1>> func1Memory_;
  std::vector<std::shared_ptr<Allocator2>> func2Memory_;
  std::vector<BufferArgPtr> func1Inputs_;
  std::vector<BufferArgPtr> func1Outputs_;
  std::vector<BufferArgPtr> func2Inputs_;
  std::vector<BufferArgPtr> func2Outputs_;
};

class CpuGpuFuncCompare
    : public Compare2Function<DEVICE_TYPE_CPU, DEVICE_TYPE_GPU> {
public:
  CpuGpuFuncCompare(const std::string& name, const FuncConfig& config)
      : Compare2Function(name + "-CPU", name + "-GPU", config) {}

  ~CpuGpuFuncCompare() {}
};

}  // namespace paddle
