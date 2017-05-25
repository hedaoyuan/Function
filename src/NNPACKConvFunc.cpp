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

#include "ConvFunc.h"
#include "nnpack.h"

DEFINE_bool(nnpack_allocate_outside,
            true,
            "Allocate and free workspace memory outside the NNPACK interface.");

namespace paddle {

nnp_convolution_algorithm get_nnp_convolution_algorithm(
    const std::string& algorithm) {
  if (algorithm == "auto") {
    return nnp_convolution_algorithm_auto;
  } else if (algorithm == "ft8x8") {
    return nnp_convolution_algorithm_ft8x8;
  } else if (algorithm == "ft16x16") {
    return nnp_convolution_algorithm_ft16x16;
  } else if (algorithm == "wt8x8") {
    return nnp_convolution_algorithm_wt8x8;
  } else if (algorithm == "implicit-gemm") {
    return nnp_convolution_algorithm_implicit_gemm;
  } else if (algorithm == "direct") {
    return nnp_convolution_algorithm_direct;
  } else {
    return nnp_convolution_algorithm_auto;
  }
}

template <DeviceType Device>
class NNPACKConvFunction : public ConvFunctionBase {
public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
    algorithm_ = get_nnp_convolution_algorithm(config.get<std::string>("algo"));
    // algorithm_ = nnp_convolution_algorithm_auto;
    transform_strategy_ = nnp_convolution_transform_strategy_compute;
    nnp_status status = nnp_initialize();
    CHECK_EQ(status, nnp_status_success);
    workspaceBuffer_ = nullptr;
    workspaceSize_ = 0;
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    check(inputs, outputs);
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);

    size_t batchSize = inputs[0].shape()[0];
    size_t inputChannels = inputs[0].shape()[1];
    size_t inputHeight = inputs[0].shape()[2];
    size_t inputWidth = inputs[0].shape()[3];
    size_t filterHeight = inputs[1].shape()[2];
    size_t filterWidth = inputs[1].shape()[2];
    size_t outputChannels = outputs[0].shape()[1];
    // size_t outputHeight = outputs[0].shape()[2];
    // size_t outputWidth = outputs[0].shape()[3];

    nnp_size inputSize = {.width = inputWidth, .height = inputHeight};
    nnp_padding padding = {.top = padding_,
                           .right = padding_,
                           .bottom = padding_,
                           .left = padding_};
    nnp_size kernelSize = {.width = filterWidth, .height = filterHeight};
    nnp_size outputSubsampling = {.width = stride_, .height = stride_};

    float* inputData = inputs[0].data<float>();
    float* filterData = inputs[1].data<float>();
    float* outputData = outputs[0].data<float>();

    void* bufferPtr = nullptr;
    size_t* sizePtr = nullptr;
    size_t needSize;
    if (FLAGS_nnpack_allocate_outside) {
      if (batchSize == 1) {
        nnp_status status = nnp_convolution_inference(
            algorithm_,
            transform_strategy_,
            inputChannels,
            outputChannels,
            inputSize,
            padding,
            kernelSize,
            outputSubsampling,
            nullptr,
            nullptr,
            nullptr, /* bias */
            nullptr,
            nullptr,
            &needSize,
            nnp_activation_identity,
            nullptr,
            nullptr, /* threadpool */
            nullptr);
        CHECK_EQ(status, nnp_status_success);
      } else {
        // only supports stride = 1
        CHECK_EQ(stride_, 1);
        nnp_status status = nnp_convolution_output(
            algorithm_,
            batchSize,
            inputChannels,
            outputChannels,
            inputSize,
            padding,
            kernelSize,
            nullptr,
            nullptr,
            nullptr, /* bias */
            nullptr,
            nullptr,
            &needSize,
            nnp_activation_identity,
            nullptr,
            nullptr, /* threadpool */
            nullptr);
        CHECK_EQ(status, nnp_status_success);
      }
  
      LOG(INFO) << "workspace size is " << needSize;
      if (needSize > workspaceSize_) {
        workspaceSize_ = needSize;
        if (workspaceBuffer_) {
          free(workspaceBuffer_);
        } else {
          posix_memalign(&workspaceBuffer_, 64, needSize);
        }
      }

      if (needSize) {
        bufferPtr = workspaceBuffer_;
        sizePtr = &needSize;
      }
    }

    if (batchSize == 1) {
      nnp_status status = nnp_convolution_inference(
          algorithm_,
          transform_strategy_,
          inputChannels,
          outputChannels,
          inputSize,
          padding,
          kernelSize,
          outputSubsampling,
          inputData,
          filterData,
          nullptr, /* bias */
          outputData,
          bufferPtr,
          sizePtr,
          nnp_activation_identity,
          nullptr,
          nullptr, /* threadpool */
          nullptr);
      CHECK_EQ(status, nnp_status_success);
    } else {
      // only supports stride = 1
      CHECK_EQ(stride_, 1);
      nnp_status status = nnp_convolution_output(
          algorithm_,
          batchSize,
          inputChannels,
          outputChannels,
          inputSize,
          padding,
          kernelSize,
          inputData,
          filterData,
          nullptr, /* bias */
          outputData,
          bufferPtr,
          sizePtr,
          nnp_activation_identity,
          nullptr,
          nullptr, /* threadpool */
          nullptr);
      CHECK_EQ(status, nnp_status_success);
    }
  }
private:
  nnp_convolution_algorithm algorithm_;
  nnp_convolution_transform_strategy transform_strategy_;
  void* workspaceBuffer_;
  size_t workspaceSize_;
};

REGISTER_TYPED_FUNC(NNPACKConv, CPU, NNPACKConvFunction);

}  // namespace paddle
