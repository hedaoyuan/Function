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

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include "unsupported/Eigen/CXX11/Tensor"
#include "MatMulFunc.h"

namespace paddle {

typedef Eigen::TensorMap<
    Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned> Matrix;

template <DeviceType Device>
class EigenMatMulFunc : public MatMulFunc {
public:
  void init(const FuncConfig& config) override {
    MatMulFunc::init(config);
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    MatMulFunc::calc(inputs, outputs);

    // Eigen::array is a std::array when c++11
    Eigen::array<int, 2> sizeA;
    sizeA[0] = inputs[0].shape()[0];
    sizeA[1] = inputs[0].shape()[1];
    Eigen::array<int, 2> sizeB;
    sizeB[0] = inputs[1].shape()[0];
    sizeB[1] = inputs[1].shape()[1];
    Eigen::array<int, 2> sizeC;
    sizeC[0] = outputs[0].shape()[0];
    sizeC[1] = outputs[0].shape()[1];

    const Matrix A(inputs[0].data<float>(), sizeA);
    const Matrix B(inputs[1].data<float>(), sizeB);
    Matrix C(outputs[0].data<float>(), sizeC);

    Eigen::DefaultDevice device;
    // Eigen::ThreadPool pool(threads);
    // Eigen::ThreadPoolDevice device(&pool, threads);

    typedef typename Eigen::Tensor<float, 2>::DimensionPair DimPair;
    Eigen::array<DimPair, 1> dims;
    dims[0] = DimPair(1, 0);
    dims[0].first = aTrans_ ? 0 : 1;
    dims[0].second = bTrans_ ? 1 : 0;

    C.device(device) = A.contract(B, dims);
  }
};

REGISTER_TYPED_FUNC(EigenMatMul, CPU, EigenMatMulFunc);
}  // namespace paddle
