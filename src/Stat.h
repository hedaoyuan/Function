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

#include <stdint.h>
#include <sys/time.h>
#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <glog/logging.h>
#include <sys/syscall.h>

#include "Locks.h"

namespace paddle {

class Stat;

inline pid_t getTID() {
  pid_t tid = syscall(__NR_gettid);
  CHECK_NE((int)tid, -1);
  return tid;
}

template <typename T>
inline std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}

class StatInfo {
public:
  explicit StatInfo() {
    total_ = 0;
    max_ = 0;
    count_ = 0;
    min_ = UINT64_MAX;
  }

  void reset() {
    total_ = 0;
    count_ = 0;
    max_ = 0;
    min_ = UINT64_MAX;
  }

  ~StatInfo() {}

  uint64_t total_;
  uint64_t max_;
  uint64_t count_;
  uint64_t min_;
};

class Stat;
typedef std::shared_ptr<Stat> StatPtr;

class StatSet {
public:
  explicit StatSet(const std::string& name) : name_(name) {}
  ~StatSet() {}

  void printAllStatus();

  StatPtr getStat(const std::string& name, pid_t tid = 0, double flops = 0) {
    std::string statName =
      name + "." + to_string(tid) + "." + to_string(flops);
    {
      ReadLockGuard guard(lock_);
      auto it = statSet_.find(statName);
      if (it != statSet_.end()) {
        return it->second;
      }
    }
    StatPtr stat = std::make_shared<Stat>(statName, tid, flops);
    std::lock_guard<RWLock> guard(lock_);
    auto ret = statSet_.insert(std::make_pair(statName, stat));
    return ret.first->second;
  }

  // reset the counters for all stats
  // clearRawData means also clearing raw tuning data, because at pserver end,
  // barrier rawData(timeVector_) is stateful, clearing it will cause rubbish
  // data, while rawData should be cleared at the new pass (so complicated
  // pserver code logic, -_- ).
  void reset(bool clearRawData = true);

private:
  std::unordered_map<std::string, StatPtr> statSet_;
  const std::string name_;
  RWLock lock_;
};

extern StatSet globalStat;

/*@brief : a simple stat*/
class Stat {
public:
  explicit Stat(const std::string& statName, pid_t tid = 0, double flops = 0)
      : name_(statName), tid_(tid), flops_(flops) {}
  ~Stat() {}

  const std::string& getName() const { return name_; }

  void addSample(uint64_t value);

  const StatInfo& getStatInfo() const { return statInfo_; };

  double getOps() const { return flops_; }

  void reset();

private:
  const std::string name_;
  pid_t tid_;
  double flops_;
  StatInfo statInfo_;
};

class TimerOnce {
public:
  TimerOnce(Stat* stat) : stat_(stat) {
    startStamp_ = nowInMicroSec();
  }
  ~TimerOnce() {
    uint64_t total = nowInMicroSec() - startStamp_;
    stat_->addSample(total);
  }

  inline uint64_t nowInMicroSec() {
    timeval tvTime;
    (void)gettimeofday(&tvTime, NULL);
    return tvTime.tv_sec * 1000000LU + tvTime.tv_usec;
  }

private:
  Stat* stat_;
  uint64_t startStamp_;
};

#ifdef PADDLE_DISABLE_TIMER

#define REGISTER_TIMER_INFO(statName)

#else

#define REGISTER_TIMER_INFO(statName, ...)                                  \
  ::paddle::StatPtr __stat =                                                \
      ::paddle::globalStat.getStat(statName, getTID(), ##__VA_ARGS__);      \
  ::paddle::TimerOnce __timerOnce(__stat.get());

#endif  // DISABLE_TIMER

}  // namespace paddle
