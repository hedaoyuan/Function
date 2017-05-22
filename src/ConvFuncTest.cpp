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

#include <gtest/gtest.h>
#include <memory>
#include "Function.h"
#include "FunctionTest.h"

namespace paddle {

TEST(Convolution, float) {
  #define NUM_LAYER 5
  size_t inputChannels[NUM_LAYER] = {3, 96, 256, 384, 384};
  size_t inputHeight[NUM_LAYER] =   {224, 27, 13, 13, 13};
  size_t inputWidth[NUM_LAYER] =    {224, 27, 13, 13, 13};
  size_t outputChannels[NUM_LAYER] = {96, 256, 384, 384, 256};

  size_t batchSize = 1;
  size_t stride = 1;
  size_t padding = 1;
  size_t filterHeight = 3;
  size_t filterWidth = 3;
  for (int i = 0; i < NUM_LAYER; i++) {
    FunctionCompare test("NaiveConvolution-CPU",
                         "ConvolutionForward-CPU",
                         FuncConfig()
                             .set("padding", padding)
                             .set("stride", stride));

    size_t outputHeight = 
        (inputHeight[i] - filterHeight + 2 *padding + stride) / stride;
    size_t outputWidth =
        (inputWidth[i] - filterWidth + 2 * padding + stride) /stride;
    TensorShape shape0{batchSize,
                       inputChannels[i],
                       inputHeight[i],
                       inputWidth[i]};
    TensorShape shape1{outputChannels[i],
                       inputChannels[i],
                       filterHeight,
                       filterHeight};
    TensorShape shape2{batchSize,
                       outputChannels[i],
                       outputHeight,
                       outputWidth};
    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape0));
    test.addInputs(BufferArg(VALUE_TYPE_FLOAT, shape1));
    test.addOutputs(BufferArg(VALUE_TYPE_FLOAT, shape2));
    test.run();
  }
}

typedef struct {
    int batchSize;
    int featureMaps;
    int imageH;
    int imageW;
    int outputFeature;
    int filter_size;
    int stride;
    int padding;
    int outputH;
    int outputW;
}_conv_parameter, *conv_parameter;
typedef std::shared_ptr<BufferArg> BufferArgPtr;

TEST(ConvolutionForward, float) {
  // 0. create function
  std::shared_ptr<FunctionBase> convGemm(
    FunctionBase::funcRegistrar_.createByType("ConvolutionForward-CPU"));
  convGemm->init(FuncConfig()
                 .set("stride", 1)
                 .set("padding", 1));

  // 1. arguments
  _conv_parameter conv_para = {batchSize      : 1,
                        featureMaps    : 3,
                        imageH         : 224,
                        imageW         : 224,
                        outputFeature  : 96,
                        filter_size    : 11,
                        stride         : 5,
                        padding        : 1,
                        outputH        : 0,
                        outputW        : 0};
  conv_para.outputH = ceil(
      (float)(conv_para.imageH - conv_para.filter_size 
              + 2*abs(conv_para.padding) + conv_para.stride) 
      / (float)conv_para.stride);
  conv_para.outputW = ceil(
      (float)(conv_para.imageW - conv_para.filter_size 
              + 2*abs(conv_para.padding) + conv_para.stride) 
      / (float)conv_para.stride);


  BufferArgs inArgs;
  BufferArgs outArgs;
  // 3. input0 Input image data
  TensorShape shape0{(size_t)conv_para.batchSize,
                    (size_t)conv_para.featureMaps,
                    (size_t)conv_para.imageH,
                    (size_t)conv_para.imageW};
  void* inputData = (void *)malloc(shape0.getElements() * sizeof(float));
  BufferArg input0(inputData, VALUE_TYPE_FLOAT, shape0);
  inArgs.addArg(input0);

  // 4. input1 Filter data
  TensorShape shape1{(size_t)conv_para.outputFeature,
                    (size_t)conv_para.featureMaps,
                    (size_t)conv_para.filter_size,
                    (size_t)conv_para.filter_size};
  void* filterData = (void *)malloc(shape1.getElements() * sizeof(float));
  BufferArg input1(filterData, VALUE_TYPE_FLOAT, shape1);
  inArgs.addArg(input1);

  // 5. output0 Output image data
  TensorShape shape2{(size_t)conv_para.batchSize,
                    (size_t)conv_para.outputFeature,
                    (size_t)conv_para.outputH,
                    (size_t)conv_para.outputW};
  void* outputData = (void *)malloc(shape2.getElements() * sizeof(float));
  BufferArg output0(outputData, VALUE_TYPE_FLOAT, shape2, ASSIGN_TO);
  outArgs.addArg(output0);

  // 6. run
  convGemm->calc(inArgs, outArgs);
}

}  // namespace paddle

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  paddle::initMain(argc, argv);
  return RUN_ALL_TESTS();
}

