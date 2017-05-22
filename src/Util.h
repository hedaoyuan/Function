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

#include <functional>

namespace paddle {

/**
 * Initialize some creators or initFunctions for layers and data
 * providers.
 * Client codes should call this function before they refer any other
 * codes that use the layer class and data provider class.
 *
 * Codes inside 'core' directory can call initMain which calls
 * runInitFunctions directly, while codes outside core can simply
 * call runInitFunctions if they don't need the commandline flags
 * designed for PADDLE main procedure.
 */
void runInitFunctions();

/**
 * Initialize logging and parse commandline
 */
void initMain(int argc, char** argv);

/**
 * Register a function, the function will be called in initMain(). Functions
 * with higher priority will be called first. The execution order of functions
 * with same priority is not defined.
 */
void registerInitFunction(std::function<void()> func, int priority = 0);
class InitFunction {
public:
  explicit InitFunction(std::function<void()> func, int priority = 0) {
    registerInitFunction(func, priority);
  }
};

}  // namespace paddle
