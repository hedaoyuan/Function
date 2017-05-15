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

#pragma once

#include <glog/logging.h>

#include "TensorShape.h"
#include "TensorType.h"

namespace paddle {

enum BufferType {
  TENSOR_UNKNOWN = 0,
  TENSOR_NORMAL = 1,
  TENSOR_SEQUENCE_ID = 2,
  TENSOR_SEQUENCE_DATA = 3,
  TENSOR_SPARSE = 4
};

enum SparseDataType {
  SPARSE_NO_VALUE = 0,  // do not need value pointer, all values are 1
  SPARSE_FLOAT_VALUE = 1
};

enum SparseDataFormat { SPARSE_CSR_FORMAT = 0, SPARSE_CSC_FORMAT = 1 };

class BufferArg;

/**
 * \brief BufferArg used as the argument type of Function.
 *
 * The arguments of the Paddle Function have four Buffer types.
 * 1. BufferArg for a dense Buffer of any dimension.
 * 2. SequenceIdArg for a Buffer of sequence start positions.
 * 3. SequenceArg for a Buffer of sequence data.
 * 4. SparseMatrixArg for a Buffer of sparse matrix.
 *
 * Buffer shape
 * For most buffers, the first dimension `shape()[0]` represents
 * the size of the mini-batch.
 *
 * Buffer argType
 * There is an ArgType property for the BufferArg used as Function Output.
 * Whether the result of the Function calculation is assigned to the
 * output Buffer or added to the output Buffer is determined by the
 * argType_ property of the output BufferArg.
 */

// ArgType is only used by output BufferArg.
// For input argument, argType_ is ignored.
// For output argument, need to set the argType_ of the BufferArg.
enum ArgType {
  UNSPECIFIED = 0,
  ASSIGN_TO = 1,
  ADD_TO = 2,
};
class BufferArg {
public:
  void setArgType(ArgType argType) { argType_ = argType; }

  ArgType getArgType() const { return argType_; }

public:
  BufferArg(ValueType valueType,
            const TensorShape& shape,
            ArgType argType = UNSPECIFIED)
      : buf_(nullptr),
        valueType_(valueType),
        shape_(shape),
        argType_(argType) {}

  BufferArg(void* buf,
            ValueType valueType,
            const TensorShape& shape,
            ArgType argType = UNSPECIFIED)
      : buf_(buf), valueType_(valueType), shape_(shape), argType_(argType) {}

  BufferArg(void* buf, ValueType valueType)
      : buf_(buf), valueType_(valueType) {}

  virtual ~BufferArg() {}

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(buf_);
  }

  void* data() const { return buf_; }
  ValueType valueType() const { return valueType_; }
  BufferType bufferType() const { return bufferType_; }
  const TensorShape& shape() const { return shape_; }

protected:
  void* buf_;
  ValueType valueType_;
  TensorShape shape_;
  BufferType bufferType_{TENSOR_UNKNOWN};
  ArgType argType_{UNSPECIFIED};
  // leading dimensions. The size is dims_.size()
  // Dims lds_;
};

}  // namespace paddle
