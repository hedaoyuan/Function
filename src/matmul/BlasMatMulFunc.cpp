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

#include "MatMulFunc.h"
#include "Gemm.h"

namespace paddle {

template <DeviceType Device>
class BlasMatMulFunc : public MatMulFunc {
public:
  void init(const FuncConfig& config) override {
    MatMulFunc::init(config);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    MatMulFunc::calc(inputs, outputs);

    const int M = outputs[0].shape()[0];
    const int N = outputs[0].shape()[1];
    const int K = !aTrans_ ? inputs[0].shape()[1] : inputs[0].shape()[0];

    gemm<float>(aTrans_ ? CblasTrans : CblasNoTrans,
                bTrans_ ? CblasTrans : CblasNoTrans,
                M,
                N,
                K,
                1.0f,
                inputs[0].data<float>(),
                aTrans_ ? M : K,
                inputs[1].data<float>(),
                bTrans_ ? K : N,
                0.0f,
                outputs[0].data<float>(),
                N);
  }
};

REGISTER_TYPED_FUNC(BlasMatMul, CPU, BlasMatMulFunc);
}  // namespace paddle
