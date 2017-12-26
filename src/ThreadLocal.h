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

#include <pthread.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <map>
#include <mutex>
#include <random>

namespace paddle {

/**
 * Thread local storage for object.
 * Example:
 *
 * Declarartion:
 * ThreadLocal<vector<int>> vec_;
 *
 * Use in thread:
 * vector<int>& vec = *vec; // obtain the thread specific object
 * vec.resize(100);
 *
 * Note that this ThreadLocal will desconstruct all internal data when thread
 * exits
 * This class is suitable for cases when frequently creating and deleting
 * threads.
 *
 * Consider implementing a new ThreadLocal if one needs to frequently create
 * both instances and threads.
 *
 * see also ThreadLocalD
 */
template <class T>
class ThreadLocal {
public:
  ThreadLocal() {
    CHECK_EQ(pthread_key_create(&threadSpecificKey_, dataDestructor), 0);
  }
  ~ThreadLocal() { pthread_key_delete(threadSpecificKey_); }

  /**
   * @brief get thread local object.
   * @param if createLocal is true and thread local object is never created,
   * return a new object. Otherwise, return nullptr.
   */
  T* get(bool createLocal = true) {
    T* p = (T*)pthread_getspecific(threadSpecificKey_);
    if (!p && createLocal) {
      p = new T();
      int ret = pthread_setspecific(threadSpecificKey_, p);
      CHECK_EQ(ret, 0);
    }
    return p;
  }

  /**
   * @brief set (overwrite) thread local object. If there is a thread local
   * object before, the previous object will be destructed before.
   *
   */
  void set(T* p) {
    if (T* q = get(false)) {
      dataDestructor(q);
    }
    CHECK_EQ(pthread_setspecific(threadSpecificKey_, p), 0);
  }

  /**
   * return reference.
   */
  T& operator*() { return *get(); }

  /**
   * Implicit conversion to T*
   */
  operator T*() { return get(); }

private:
  static void dataDestructor(void* p) { delete (T*)p; }

  pthread_key_t threadSpecificKey_;
};

}  // namespace paddle
