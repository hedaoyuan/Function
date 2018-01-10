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

#include <algorithm>
#include "ConvOp.h"
#include "GemmFunctor.h"
#include "Im2Col.h"
#include "Stat.h"

namespace paddle {

/**
 * Convolution calculation based on matrix multiplication.
 *
 * Convolution calculation based on matrix multiplication consists of two steps.
 * 1. Unrolling the input data in convolution order into a two-dimensional
 *    convolution matrix.
 * 2. Convolution results are obtained by matrix multiplication based on
 *    filter matrix and the convolution matrix.
 * 
 */
template <class T, class Im2ColFunctor, class GemmFunctor>
struct ConvUsingGemm {
  static void compute(const PackingParameter& argument,
                      const T* input,
                      const T* filter,
                      int filter_ld,
                      T* output,
                      int output_ld,
                      T* packed_input,
                      int input_channels,
                      int output_channels,
                      int block_channels,
                      int block_output_height,
                      T beta) {
    const int input_offset = argument.input_height * argument.input_width;
    const int filter_offset = argument.filter_height * argument.filter_width;
    const int output_width = argument.output_width;
    const int output_height = argument.output_height;

    Im2ColFunctor im2col;
    GemmFunctor gemm;
    for (int ic = 0; ic < input_channels; ic += block_channels) {
      int packed_channels = std::min(input_channels - ic, block_channels);
      for (int oh = 0; oh < output_height; oh += block_output_height) {
        int packed_output_height = std::min(output_height - oh, block_output_height);

        int M = output_channels;
        int N = packed_output_height * output_width;
        int K = packed_channels * filter_offset;
        {
        REGISTER_TIMER_INFO("im2col");
        im2col(argument,
               input + ic * input_offset,
               packed_input,
               packed_channels,
               oh,
               packed_output_height,
               N);
        }
        {
        REGISTER_TIMER_INFO("gemm");
        gemm(CblasNoTrans,
             CblasNoTrans,
             M,
             N,
             K,
             1.0f,
             filter + ic * filter_offset,
             filter_ld,
             packed_input,
             N,
             beta,
             output + oh * output_width,
             output_ld);
        }
      }
      // The input beta can be 0.0;
      beta = 1.0;
    }
  }
};

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

    float beta;
    if (outputs[0].getArgType() == ADD_TO) {
      beta = 1.0;
    } else {
      beta = 0.0;
    }

    size_t batch_size = input[0];
    size_t input_channels = input[1];
    size_t input_height = input[2];
    size_t input_width = input[3];
    size_t filter_height = getFilterHeight(filter);
    size_t filter_width = getFilterWidth(filter);
    size_t output_channels = output[1];
    size_t output_height = output[2];
    size_t output_width = output[3];

    float* input_data = inputs[0].data<float>();
    float* output_data = outputs[0].data<float>();
    float* packed_input = NULL;
    bool needIm2col = isNeedIm2col(filter);

    // size_t block_output_height = 
    //    std::min(outputWidth > 2048 ? 1 : 2048 / outputWidth, outputHeight);
    // Max col matrix width 2048, Max col matrix size 2M.
    size_t block_output_height =
      std::min(std::max(2048 / output_width, (size_t)1), output_height);
    size_t maxColWidth = block_output_height * output_width;
    size_t block_channels =
      std::min(std::max((524288 / maxColWidth) / filter_height * filter_width, (size_t)1),
               input_channels / groups_);
    size_t maxColHeight = block_channels * filter_height * filter_width;

    PackingParameter packing_argument;
    if (needIm2col) {
      packing_argument.input_height = input_height;
      packing_argument.input_width = input_width;
      packing_argument.filter_height = filter_height;
      packing_argument.filter_width = filter_width;
      packing_argument.output_height = output_height;
      packing_argument.output_width = output_width;
      packing_argument.stride_height = strideH();
      packing_argument.stride_width = strideW();
      packing_argument.padding_height = paddingH();
      packing_argument.padding_width = paddingW();
      resizeBuffer<Device>(maxColHeight * maxColWidth * sizeof(float));
      packed_input = reinterpret_cast<float*>(memory_->getBuf());
    }

    GemmFunctor<Device, float> gemm;
    size_t input_offset = (input_channels / groups_) * input_height * input_width;
    size_t output_offset =
        (output_channels / groups_) * output_height * output_width;
    size_t filter_offset = filter.getElements() / groups_;

    for (size_t i = 0; i < batch_size; i++) {
      float* filter_data = inputs[1].data<float>();
      for (size_t g = 0; g < groups_; g++) {
        if (needIm2col) {
          ConvUsingGemm<float,
                        Im2ColFunctor<kCFO, Device, float>,
                        GemmFunctor<Device, float>>::compute(
                          packing_argument,
                          input_data,
                          filter_data,
                          input_channels / groups_ * filter_height * filter_width,
                          output_data,
                          output_height * output_width,
                          packed_input,
                          input_channels / groups_,
                          output_channels / groups_,
                          block_channels,
                          block_output_height,
                          beta);
        } else {
          int M = output_channels / groups_;
          int N = output_height * output_width;
          int K = input_channels / groups_ * filter_height * filter_width;
          gemm(CblasNoTrans,
               CblasNoTrans,
               M,
               N,
               K,
               1.0f,
               filter_data,
               K,
               input_data,
               N,
               beta,
               output_data,
               N);
        }
        input_data += input_offset;
        output_data += output_offset;
        filter_data += filter_offset;
      }
    }
  }
};

REGISTER_TYPED_FUNC(GemmConv, CPU, GemmConvFunction);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(GemmConv, GPU, GemmConvFunction);
#endif

}  // namespace paddle
