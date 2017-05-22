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

#include <mkl.h>

namespace paddle {

template <class T>
void gemm(const CBLAS_TRANSPOSE transA,
          const CBLAS_TRANSPOSE transB,
          const int M,
          const int N,
          const int K,
          const T alpha,
          const T* A,
          const int lda,
          const T* B,
          const int ldb,
          const T beta,
          T* C,
          const int ldc);

}  // namespace paddle
