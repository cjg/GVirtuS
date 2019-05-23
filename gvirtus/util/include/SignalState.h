#ifndef GVIRTUS_SIGSTATE_H
#define GVIRTUS_SIGSTATE_H

#include <functional>
#include <map>

namespace gvirtus::util {
  class SignalState {
  public:
    SignalState() = default;
    ~SignalState() = default;
    void setup_signal_state(int);

    static inline bool
    get_signal_state(int signo) {
      return _signals_state[signo];
    }

  private:
    static void true_flag(int signo) {
      _signals_state[signo] = true;
    }

    static std::map<int, bool> _signals_state;
  };

} // namespace gvirtus::util

#endif // GVIRTUS_SIGSTATE_H