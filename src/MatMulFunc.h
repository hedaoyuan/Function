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

namespace paddle {

class MatMulFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    aTrans_ = config.get<bool>("aTrans");
    bTrans_ = config.get<bool>("bTrans");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK(!aTrans_ || !bTrans_)
        << "Not support both a and b are transpose matrices";

    CHECK_EQ((size_t)2, inputs.size());
    CHECK_EQ((size_t)1, outputs.size());
    CHECK_EQ(inputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[1].shape().ndims(), (size_t)2);
    CHECK_EQ(outputs[0].shape().ndims(), (size_t)2);

    size_t aRow = !aTrans_ ? inputs[0].shape()[0] : inputs[0].shape()[1];
    size_t aCol = !aTrans_ ? inputs[0].shape()[1] : inputs[0].shape()[0];
    size_t bRow = !bTrans_ ? inputs[1].shape()[0] : inputs[1].shape()[1];
    size_t bCol = !bTrans_ ? inputs[1].shape()[1] : inputs[1].shape()[0];
    CHECK_EQ(aCol, bRow);
    CHECK_EQ(aRow, outputs[0].shape()[0]);
    CHECK_EQ(bCol, outputs[0].shape()[1]);
  }

  int64_t ops(const BufferArgs& inputs, const BufferArgs& outputs) override {
    const int64_t M = outputs[0].shape()[0];
    const int64_t N = outputs[0].shape()[1];
    const int64_t K = !aTrans_ ? inputs[0].shape()[1] : inputs[0].shape()[0];

    // number of floating-point operations
    int64_t ops = 2 * M * N * K;

    return ops;
  }

protected:
  bool aTrans_;
  bool bTrans_;
};

}  // namespace paddle
