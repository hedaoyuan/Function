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

#include "Stat.h"
#include <algorithm>
#include <iomanip>
#include "Util.h"

namespace paddle {

StatSet globalStat("GlobalStatInfo");

void Stat::addSample(uint64_t value) {
  if (value > statInfo_.max_) {
    statInfo_.max_ = value;
  }
  if (value < statInfo_.min_) {
    statInfo_.min_ = value;
  }
  statInfo_.total_ += value;
  statInfo_.count_++;
}

void Stat::reset() {
  statInfo_.reset();
}

std::ostream& operator<<(std::ostream& outPut, const Stat& stat) {
  const StatInfo& info = stat.getStatInfo();

  uint64_t average = 0;
  double ops = stat.getOps();
  if (info.count_ > 0) {
    outPut << std::setfill(' ') << std::left;
    average = info.total_ / info.count_;
    outPut << "Stat=" << std::setw(30) << stat.getName();

    if (ops > 0) {
      double gflops = ops / average / 1000;
      outPut << " gflops=" << std::setw(10) << gflops;
    }
    outPut << " total=" << std::setw(10) << info.total_ * 0.001
           << " avg=" << std::setw(10) << average * 0.001
           << " max=" << std::setw(10) << info.max_ * 0.001
           << " min=" << std::setw(10) << info.min_ * 0.001
           << " count=" << std::setw(10) << info.count_ << std::endl;
  }

  return outPut;
}

void StatSet::printAllStatus() {
  ReadLockGuard guard(lock_);
  LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
            << "======= StatSet: [" << name_ << "] status ======" << std::endl;
  for (auto& stat : statSet_) {
    if (stat.second->getStatInfo().count_ > 0) {
      LOG(INFO) << std::setiosflags(std::ios::left) << std::setfill(' ')
                << *(stat.second);
    }
  }
  LOG(INFO) << std::setiosflags(std::ios::left)
            << "--------------------------------------------------"
            << std::endl;
}

void StatSet::reset(bool clearRawData) {
  ReadLockGuard guard(lock_);
  for (auto& stat : statSet_) {
    stat.second->reset();
  }
}
}  // namespace paddle
