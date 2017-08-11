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

#include <malloc.h>
#include <memory>

namespace paddle {

class MemoryHandle {
protected:
  explicit MemoryHandle(size_t size) : size_(size), buf_(nullptr) {}
  virtual ~MemoryHandle() {}

public:
  void* getBuf() const { return buf_; }
  size_t getSize() const { return size_; }

protected:
  // address returned by GPU/CPU memory allocation
  size_t size_;
  void* buf_;
};

/**
 * Wrapper class for raw cpu memory handle.
 * The raw handle will be released at destructor
 */
class CpuMemoryHandle : public MemoryHandle {
public:
  explicit CpuMemoryHandle(size_t size) : MemoryHandle(size) {
    buf_ = memalign(32ul, size);
    CHECK(buf_) << "Fail to allocate CPU memory: size=" << size;
    if (size > 1000 * 1000 * 1000) {
      LOG(INFO) << "CpuMemoryHandle alloc big buf size=" << size;
    }
  }
  virtual ~CpuMemoryHandle() {
    if (buf_) {
      free(buf_);
    }
  }
};

/**
 * Wrapper class for raw cpu memory handle.
 *
 * The raw handle will be released at destructor
 */
class GpuMemoryHandle : public MemoryHandle {
public:
  explicit GpuMemoryHandle(size_t size) : MemoryHandle(size) {}
  virtual ~GpuMemoryHandle() {}
};

typedef std::shared_ptr<MemoryHandle> MemoryHandlePtr;
typedef std::shared_ptr<CpuMemoryHandle> CpuMemHandlePtr;
typedef std::shared_ptr<GpuMemoryHandle> GpuMemHandlePtr;
}  // namespace paddle
