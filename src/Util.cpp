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

#include "Util.h"
#include <vector>
#include <mutex>
#include <algorithm>
#include <glog/logging.h>

namespace paddle {

static bool g_initialized = false;
typedef std::pair<int, std::function<void()>> PriorityFuncPair;
typedef std::vector<PriorityFuncPair> InitFuncList;
static InitFuncList* g_initFuncs = nullptr;
static std::once_flag g_onceFlag;
void registerInitFunction(std::function<void()> func, int priority) {
  if (g_initialized) {
    LOG(FATAL) << "registerInitFunction() should only called before initMain()";
  }
  if (!g_initFuncs) {
    g_initFuncs = new InitFuncList();
  }
  g_initFuncs->push_back(std::make_pair(priority, func));
}

void runInitFunctions() {
  std::call_once(g_onceFlag, []() {
    VLOG(3) << "Calling runInitFunctions";
    if (g_initFuncs) {
      std::sort(g_initFuncs->begin(),
                g_initFuncs->end(),
                [](const PriorityFuncPair& x, const PriorityFuncPair& y) {
                  return x.first > y.first;
                });
      for (auto& f : *g_initFuncs) {
        f.second();
      }
      delete g_initFuncs;
      g_initFuncs = nullptr;
    }
    g_initialized = true;
    VLOG(3) << "Call runInitFunctions done.";
  });
}

void initMain(int argc, char** argv) {
  runInitFunctions();
}

}  // namespace paddle
