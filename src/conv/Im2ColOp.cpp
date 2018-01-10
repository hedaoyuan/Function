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

#include "Im2Col.h"

namespace paddle {

template <class T>
class Im2ColFunctor<kCFO, DEVICE_TYPE_CPU, T> {
public:
  void operator()(const PackingParameter& argument,
                  const T* input,
                  T* packed_input,
                  int packed_channels,
                  int packed_output_start,
                  int packed_output_height,
                  int packed_ld) {
    const int input_height = argument.input_height;
    const int input_width = argument.input_width;
    const int filter_height = argument.filter_height;
    const int filter_width = argument.filter_width;
    const int output_width = argument.output_width;
    const int stride_height = argument.stride_height;
    const int stride_width = argument.stride_width;
    const int padding_height = argument.padding_height;
    const int padding_width = argument.padding_width;

    if (stride_height == 1 && stride_width == 1) {
      // No Padding
      if (padding_height == 0 && padding_width == 0) {
        for (int ic = 0; ic < packed_channels; ic++) {
          for (int oh = 0; oh < packed_output_height; oh++) {
            const T* src_data = input +
              (oh + packed_output_start) * input_width;
            T* dst_data = packed_input + oh * output_width;
            for (int fh = 0; fh < filter_height; fh++) {
              for (int fw = 0; fw < filter_width; fw++) {
                memcpy(dst_data, src_data + fw, output_width * sizeof(T));
                dst_data += packed_ld;
              }
              src_data += input_width;
            }
          }
          packed_input += filter_height * filter_width * packed_ld;
          input += input_height * input_width;
        }
      }

      if (padding_height == 1 && padding_width == 1) {
        for (int ic = 0; ic < packed_channels; ic++) {
          for (int oh = 0; oh < packed_output_height; oh++) {
            int input_row_offset = oh + packed_output_start - padding_height;
            const T* src_data = input +
              (input_row_offset > 0 ? input_row_offset : 0) * input_width;
            T* dst_data = packed_input + oh * output_width;
            for (int fh = 0; fh < filter_height; fh++) {
              for (int fw = 0; fw < filter_width; fw++) {
                if (fh + input_row_offset < 0 ||
                    fh + input_row_offset >= input_height) {
                  memset(dst_data, 0, output_width * sizeof(T));
                } else {
                  if (fw < padding_width) {
                    int padding_size = padding_width - fw;
                    for (int n = 0; n < padding_size; n++) {
                      dst_data[n] = (float)0.0f;
                    }
                    memcpy(dst_data + padding_size,
                           src_data,
                           (output_width - padding_size) * sizeof(T));
                  } else if (fw - padding_width + output_width > input_width) {
                    int padding_size = padding_width - filter_width + 1 + fw;
                    memcpy(dst_data,
                           src_data + fw - padding_width,
                           (output_width - padding_size) * sizeof(T));
                    for (int n = 0; n < padding_size; n++) {
                      dst_data[output_width - padding_size + n] = (float)0.0f;
                    }
                  } else {
                    memcpy(dst_data,
                           src_data + fw - padding_width,
                           output_width * sizeof(T));
                  }
                }
                dst_data += packed_ld;
              }

              if (fh + input_row_offset >= 0 &&
                  fh + input_row_offset < input_height) {
                src_data += input_width;
              }
            }
          }
          packed_input += filter_height * filter_width * packed_ld;
          input += input_height * input_width;
        }
      }

      return;
    }

    for (int ic = 0; ic < packed_channels; ic++) {
      for (int oh = 0; oh < packed_output_height; oh++) {
        T* dst_data = packed_input + oh * output_width;
        for (int fh = 0; fh < filter_height; fh++) {
          for (int fw = 0; fw < filter_width; fw++) {
            int imRowIdx =
              (oh + packed_output_start) * stride_height + fh - padding_height;
            if (imRowIdx < 0 || imRowIdx >= input_height) {
              memset(dst_data, 0, output_width * sizeof(T));
            } else {
              for (int ow = 0; ow < output_width; ow++) {
                int imColIdx = ow * stride_width + fw - padding_width;
                if (imColIdx < 0 || imColIdx >= input_width) {
                  dst_data[ow] = T(0);
                } else {
                  dst_data[ow] = input[imRowIdx * input_width + imColIdx];
                }
              }
            }
            dst_data += packed_ld;
          }
        }
      }
      packed_input += filter_height * filter_width * packed_ld;
      input += input_height * input_width;
    }
  }
};

template class Im2ColFunctor<kCFO, DEVICE_TYPE_CPU, float>;

#if 0

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [inputChannels, filterHeight, filterWidth, outputHeight, outputWidth]
 */
template <class T>
class Col2ImFunctor<kCFO, DEVICE_TYPE_CPU, T> {
public:
  void operator()(T* imData,
                  const TensorShape& imShape,
                  const T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[1];
    int filterWidth = colShape[2];
    int outputHeight = colShape[3];
    int outputWidth = colShape[4];
    int channelsCol = inputChannels * filterHeight * filterWidth;

    for (int c = 0; c < channelsCol; ++c) {
      int wOffset = c % filterWidth;
      int hOffset = (c / filterWidth) % filterHeight;
      int c_im = c / filterWidth / filterHeight;
      for (int h = 0; h < outputHeight; ++h) {
        for (int w = 0; w < outputWidth; ++w) {
          int imRowIdx = h * strideHeight + hOffset;
          int imColIdx = w * strideWidth + wOffset;
          if ((imRowIdx - paddingHeight) >= 0 &&
              (imRowIdx - paddingHeight) < inputHeight &&
              (imColIdx - paddingWidth) >= 0 &&
              (imColIdx - paddingWidth) < inputWidth) {
            imRowIdx += c_im * inputHeight - paddingHeight;
            imColIdx -= paddingWidth;
            imData[imRowIdx * inputWidth + imColIdx] +=
                colData[(c * outputHeight + h) * outputWidth + w];
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<kCFO, DEVICE_TYPE_CPU, float>;
template class Im2ColFunctor<kCFO, DEVICE_TYPE_CPU, double>;
template class Col2ImFunctor<kCFO, DEVICE_TYPE_CPU, float>;
template class Col2ImFunctor<kCFO, DEVICE_TYPE_CPU, double>;

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [outputHeight, outputWidth, inputChannels, filterHeight, filterWidth]
 */
template <class T>
class Im2ColFunctor<kOCF, DEVICE_TYPE_CPU, T> {
public:
  void operator()(const T* imData,
                  const TensorShape& imShape,
                  T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[3];
    int filterWidth = colShape[4];
    int outputHeight = colShape[0];
    int outputWidth = colShape[1];
    for (int outputH = 0; outputH < outputHeight; ++outputH) {
      int imRowOffset = outputH * strideHeight - paddingHeight;
      for (int outputW = 0; outputW < outputWidth; ++outputW) {
      int imColOffset = outputW * strideWidth  - paddingWidth;
        for (int channel = 0; channel < inputChannels; ++channel) {
          for (int filterH = 0; filterH < filterHeight; ++filterH) {
            for (int filterW = 0; filterW < filterWidth; ++filterW) {
              int colDataOffset =
                  (((outputH * outputWidth + outputW) * inputChannels +
                    channel) *
                       filterHeight +
                   filterH) *
                      filterWidth +
                  filterW;
              if (imRowOffset + filterH < 0 || imRowOffset + filterH >= inputHeight ||
                  imColOffset + filterW < 0 || imColOffset + filterW >= inputWidth) {
                colData[colDataOffset] = float(0);
              } else {
                int imDataOffset =
                    (channel * inputHeight + imRowOffset + filterH) * inputWidth +
                    imColOffset + filterW;
                colData[colDataOffset] = imData[imDataOffset];
              }
            }
          }
        }
      }
    }
  }
};

/*
 * imShape = [inputChannels, inputHeight, inputWidth]
 * colShape =
 *   [outputHeight, outputWidth, inputChannels, filterHeight, filterWidth]
 */
template <class T>
class Col2ImFunctor<kOCF, DEVICE_TYPE_CPU, T> {
public:
  void operator()(T* imData,
                  const TensorShape& imShape,
                  const T* colData,
                  const TensorShape& colShape,
                  int strideHeight,
                  int strideWidth,
                  int paddingHeight,
                  int paddingWidth) {
    int inputChannels = imShape[0];
    int inputHeight = imShape[1];
    int inputWidth = imShape[2];
    int filterHeight = colShape[3];
    int filterWidth = colShape[4];
    int outputHeight = colShape[0];
    int outputWidth = colShape[1];
    for (int outputH = 0; outputH < outputHeight; ++outputH) {
      for (int outputW = 0; outputW < outputWidth; ++outputW) {
        for (int channel = 0; channel < inputChannels; ++channel) {
          for (int filterH = 0; filterH < filterHeight; ++filterH) {
            for (int filterW = 0; filterW < filterWidth; ++filterW) {
              int imRowOffset =
                  outputH * strideHeight + filterH - paddingHeight;
              int imColOffset = outputW * strideWidth + filterW - paddingWidth;
              int colDataOffset =
                  (((outputH * outputWidth + outputW) * inputChannels +
                    channel) *
                       filterHeight +
                   filterH) *
                      filterWidth +
                  filterW;
              if (imRowOffset >= 0 && imRowOffset < inputHeight &&
                  imColOffset >= 0 && imColOffset < inputWidth) {
                int imDataOffset =
                    (channel * inputHeight + imRowOffset) * inputWidth +
                    imColOffset;
                imData[imDataOffset] += colData[colDataOffset];
              }
            }
          }
        }
      }
    }
  }
};

template class Im2ColFunctor<kOCF, DEVICE_TYPE_CPU, float>;
template class Im2ColFunctor<kOCF, DEVICE_TYPE_CPU, double>;
template class Col2ImFunctor<kOCF, DEVICE_TYPE_CPU, float>;
template class Col2ImFunctor<kOCF, DEVICE_TYPE_CPU, double>;

#endif

}  // namespace paddle
