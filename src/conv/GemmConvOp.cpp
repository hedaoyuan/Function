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

    if (needIm2col) {
      colShape = TensorShape({inputChannels / groups_,
                              filterHeight,
                              filterWidth,
                              outputHeight,
                              outputWidth});
      resizeBuffer<Device>(colShape.getElements());
      colData = reinterpret_cast<real*>(memory_->getBuf());
    }

    Im2ColFunctor<kCFO, Device, real> im2col;
    GemmFunctor<Device, real> gemm;
    size_t inputOffset = imShape.getElements();
    size_t outputOffset =
        (outputChannels / groups_) * outputHeight * outputWidth;
    size_t filterOffset = filter.getElements() / groups_;

    for (size_t i = 0; i < batchSize; i++) {
      for (size_t g = 0; g < groups_; g++) {
        if (needIm2col) {
          im2col(inputData + g * inputOffset,
                 imShape,
                 colData,
                 colShape,
                 strideH(),
                 strideW(),
                 paddingH(),
                 paddingW());
        } else {
          colData = inputData + g * inputOffset;
        }
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
             colData,
             N,
             beta,
             outputData + g * outputOffset,
             N);
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
