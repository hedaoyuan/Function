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

#include "ConvOp.h"
#include "GemmFunctor.h"
#include "Im2Col.h"
#include <cmath>
#include "Stat.h"

namespace paddle {

/*
 * \brief Forward calculation of convolution.
 */
template <DeviceType Device>
class GemmConvFunction : public ConvFunctionBase {
public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
  }

  void check(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();
    checkShape(input, filter, output);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    REGISTER_TIMER_INFO("GemmConv", "GemmConv");
    CHECK_EQ(numInputs_, inputs.size());
    CHECK_EQ(numOutputs_, outputs.size());
    check(inputs, outputs);
    // TODO(hedaoyuan): Need to define some index macros,
    // to avoid useing 0 and 1.
    const TensorShape& input = inputs[0].shape();
    const TensorShape& filter = inputs[1].shape();
    const TensorShape& output = outputs[0].shape();

    real beta;
    if (outputs[0].getArgType() == ADD_TO) {
      beta = 1.0;
    } else {
      beta = 0.0;
    }

    size_t batchSize = input[0];
    size_t inputChannels = input[1];
    size_t inputHeight = input[2];
    size_t inputWidth = input[3];
    size_t filterHeight = getFilterHeight(filter);
    size_t filterWidth = getFilterWidth(filter);
    size_t outputChannels = output[1];
    size_t outputHeight = output[2];
    size_t outputWidth = output[3];

    real* inputData = inputs[0].data<real>();
    real* filterData = inputs[1].data<real>();
    real* outputData = outputs[0].data<real>();
    bool needIm2col = isNeedIm2col(filter);

    TensorShape imShape =
        TensorShape({inputChannels / groups_, inputHeight, inputWidth});

    TensorShape colShape;
    real* colData = NULL;

    size_t colHeight = inputChannels / groups_ * filterHeight * filterWidth;
    size_t colWidth = outputHeight * outputWidth;
    // Max col matrix height 256, Max col matrix width 1024
    size_t stepColHeight = std::min(colHeight, (size_t)256);
    size_t stepColWidth = std::min(colWidth, (size_t)1024);

    if (needIm2col) {
      colShape = TensorShape({inputChannels / groups_,
                              filterHeight,
                              filterWidth,
                              outputHeight,
                              outputWidth});

      resizeBuffer<Device>(stepColHeight * stepColWidth * sizeof(real));
      colData = reinterpret_cast<real*>(memory_->getBuf());
    }

    Im2ColFunctor<kCFO, Device, real> im2col;
    GemmFunctor<Device, real> gemm;
    size_t inputOffset = imShape.getElements();
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;

    int nStride = colWidth;
    int kStride = colHeight;
    for (size_t i = 0; i < batchSize; i++) {
      for (size_t g = 0; g < groups_; g++) {
        if (needIm2col) {
          real beta_ = beta;
          for (size_t colHeightStart = 0; colHeightStart < colHeight; colHeightStart += stepColHeight) {
            for (size_t colWidthStart = 0; colWidthStart < colWidth; colWidthStart += stepColWidth) {
              int N = std::min(colWidth - colWidthStart, stepColWidth);
              int K = std::min(colHeight - colHeightStart, stepColHeight);
              // im2col
              {
              REGISTER_TIMER_INFO("im2col", "GemmConv");
              im2col(inputData + g * inputOffset,
                     imShape,
                     colData,
                     colShape,
                     strideH(),
                     strideW(),
                     paddingH(),
                     paddingW(),
                     colHeightStart,
                     K,
                     colWidthStart,
                     N);
              }

              // gemm
              {
              REGISTER_TIMER_INFO("gemm", "GemmConv");
              int M = outputChannels / groups_;
              gemm(CblasNoTrans,
                   CblasNoTrans,
                   M,
                   N,
                   K,
                   1.0f,
                   filterData + g * filterOffset + colHeightStart,
                   kStride,
                   colData,
                   N,
                   beta_,
                   outputData + g * outputOffset + colWidthStart,
                   nStride);
              }
            }
            beta_ = 1.0;
          }
        } else {
          int M = outputChannels / groups_;
          int N = outputHeight * outputWidth;
          int K = inputChannels / groups_ * filterHeight * filterWidth;
          gemm(CblasNoTrans,
               CblasNoTrans,
               M,
               N,
               K,
               1.0f,
               filterData + g * filterOffset,
               K,
               inputData + g * inputOffset,
               N,
               beta,
               outputData + g * outputOffset,
               N);
        }
      }
      inputData += inputChannels * inputHeight * inputWidth;
      outputData += outputChannels * outputHeight * outputWidth;
    }
  }
};

REGISTER_TYPED_FUNC(GemmConv, CPU, GemmConvFunction);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(GemmConv, GPU, GemmConvFunction);
#endif

}  // namespace paddle
