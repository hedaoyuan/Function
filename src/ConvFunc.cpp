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
#include "Gemm.h"

namespace paddle {

/*
 * inputData  = [batchSize, inputChannels, inputHeight, inputWidth]
 * filterData = [outputChannels, inputChannels, filterHeight, filterWidth]
 * outputData = [batchSize, outputChannels, outputHeight, outputWidth]
 * The three arguments are stored in memory in row major order.
 */
template <class T>
class NaiveConvFunctor {
public:
  void operator()(const T* inputData,
                  size_t batchSize,
                  size_t inputChannels,
                  size_t inputHeight,
                  size_t inputWidth,
                  const T* filterData,
                  size_t filterHeight,
                  size_t filterWidth,
                  T* outputData,
                  size_t outputChannels,
                  size_t outputHeight,
                  size_t outputWidth,
                  size_t padding,
                  size_t stride) {
    for(size_t batch = 0; batch < batchSize; batch++) {
      for (size_t outC = 0; outC < outputChannels; outC++) {
        for (size_t outH = 0; outH < outputHeight; outH++) {
          for (size_t outW = 0; outW < outputWidth; outW++) {
            const int inStartH = (outH * stride) - padding;
            const int inStartW = (outW * stride) - padding;
            T outValue = (T)0;
            for (size_t inC = 0; inC < inputChannels; inC++) {
              for (size_t fH = 0; fH < filterHeight; fH++) {
                for (size_t fW = 0; fW < filterWidth; fW++) {
                  T inValue;
                  const int inH = inStartH + fH;
                  const int inW = inStartW + fW;
                  if ((inH >= 0 && inH < inputHeight) &&
                      (inW >= 0 && inW < inputWidth)) {
                    size_t offsetInput = 
                      batch * inputChannels * inputHeight * inputWidth
                      + inC * inputHeight * inputWidth
                      + inH * inputWidth
                      + inW;
                    inValue = inputData[offsetInput];
                  } else {
                    inValue = (T)0;
                  }
                  size_t offsetFilter =
                    outC * inputChannels * filterHeight * filterWidth
                    + inC * filterHeight * filterWidth
                    + fH * filterWidth
                    + fW;
                  T filterValue = filterData[offsetFilter];
                  outValue += (inValue * filterValue);
                }
              }
            }

            size_t offset =
                batch * outputChannels * outputHeight * outputWidth
                + outC * outputHeight * outputWidth
                + outH * outputWidth
                + outW;
            outputData[offset] = outValue;
          }
        }
      }
    }
  }
};

template <DeviceType Device>
class NaiveConvFunction : public ConvFunctionBase {
public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
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
    size_t outputHeight = outputs[0].shape()[2];
    size_t outputWidth = outputs[0].shape()[3];

    float* inputData = inputs[0].data<float>();
    float* filterData = inputs[1].data<float>();
    float* outputData = outputs[0].data<float>();
    NaiveConvFunctor<float> conv;
    conv(inputData,
         batchSize,
         inputChannels,
         inputHeight,
         inputWidth,
         filterData,
         filterHeight,
         filterWidth,
         outputData,
         outputChannels,
         outputHeight,
         outputWidth,
         padding_,
         stride_);
  }
};

REGISTER_TYPED_FUNC(NaiveConvolution, CPU, NaiveConvFunction);

/*
 * input_data = [batche_size, input_channels, input_height, input_width]
 * filter_data = [output_channels, input_channels, filter_height, filter_width]
 * output_data = [batche_size, output_channels, output_height, output_width]
 * 
 *
 * output[H = output_channels, W = output_height*output_width] 
 *  = filter[H = output_channels, W = input_channels*filter_height*filter_width]
 *    * im2col[H = input_channels*filter_height*filter_width,
 *             W = output_height*output_width]
 *
 *
 */


/*
 *
 * imData = [input_channels, input_height, input_width]
 * colData = [input_channels, filter_height, filter_width, output_height, output_width]
 *
 */
template<class T>
class Im2ColFunctor {
public:
  void operator()(const T* imData,
                  int inputChannels,
                  int inputHeight,
                  int inputWidth,
                  int filterHeight,
                  int filterWidth,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth,
                  int outputHeight,
                  int outputWidth,
                  T* colData) {
    int channelsCol = inputChannels * filterHeight * filterWidth;

    for (int c = 0; c < channelsCol; ++c) {
      int wOffset = c % filterWidth;
      int hOffset = (c / filterWidth) % filterHeight;
      int c_im = c / filterHeight / filterWidth;
      for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
          // no c_im*height to Exclude the channel number
          int imgRowIdx = h * strideHeight + hOffset;
          int imgColIdx = w * strideWidth + wOffset;
          if ((imgRowIdx - paddingHeight) < 0 ||
              (imgRowIdx - paddingHeight) >= inputHeight ||
              (imgColIdx - paddingWidth) < 0 ||
              (imgColIdx - paddingWidth) >= inputWidth) {
            colData[(c * outputHeight + h) * outputWidth + w] = T(0);
          } else {
            imgRowIdx += c_im * inputHeight - paddingHeight;
            imgColIdx -= paddingWidth;
            colData[(c * outputHeight + h) * outputWidth + w] =
                imData[imgRowIdx * inputWidth + imgColIdx];
          }
        }
      }
    }
  }
};

/*
 * Function Arguments:
 *
 * \param inputs[0]  Input image data, is NCHW format, where N is batch size,
 *                   C is the number of channels, H and W is the height and
 *                   width of input image.
 * \param inputs[1]  Filter data, is MCHW, where M is the number of output
 *                   channels, C is the number of input channels, H and W
 *                   is height and width of filter.
 * \param outputs[0] Output image data, is NCHW format, where N is batch size,
  *                  C is the number of channels, H and W is the height and
 *                   width of output image.
 */
template <DeviceType Device>
class GemmConvFunction : public ConvFunctionBase {
public:
  void init(const FuncConfig& config) override {
    ConvFunctionBase::init(config);
    colData = nullptr;
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
    size_t outputHeight = outputs[0].shape()[2];
    size_t outputWidth = outputs[0].shape()[3];

    float* inputData = inputs[0].data<float>();
    float* filterData = inputs[1].data<float>();
    float* outputData = outputs[0].data<float>();
    Im2ColFunctor<float> im2col;
    if (colData == nullptr) {
      size_t size = inputChannels * filterHeight * filterWidth * outputHeight * outputWidth;
      colData = (float*)malloc(size * sizeof(float));
    }

    for (size_t i = 0; i < batchSize; i++) {
      inputData += i * inputChannels * inputHeight * inputWidth;
      outputData += i * outputChannels * outputHeight * outputWidth;
      im2col(inputData,
             inputChannels,
             inputHeight,
             inputWidth,
             filterHeight,
             filterWidth,
             stride_,
             stride_,
             padding_,
             padding_,
             outputHeight,
             outputWidth,
             colData);

      int M = outputChannels;
      int N = outputHeight*outputWidth;
      int K = inputChannels*filterHeight*filterWidth;
      gemm<float>(CblasNoTrans,
                 CblasNoTrans,
                 M,
                 N,
                 K,
                 1.0f,
                 filterData,
                 K,
                 colData,
                 N,
                 1.0f,
                 outputData,
                 N);
    }
  }

private:
  float* colData;
};


REGISTER_TYPED_FUNC(ConvolutionForward, CPU, GemmConvFunction);

}  // namespace paddle
