#pragma once

#include <functional>
#include <map>

namespace gvirtus::common {
class SignalState {
 public:
  SignalState() = default;
  ~SignalState() = default;
  void setup_signal_state(int);

  static inline bool get_signal_state(int signo) {
    return _signals_state[signo];
  }
 private:
  static void true_flag(int signo) { _signals_state[signo] = true; }
  static std::map<int, bool> _signals_state;
};
}
