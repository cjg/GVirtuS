#ifndef GVIRTUS_SIGEXCEPT_H
#define GVIRTUS_SIGEXCEPT_H

#include <stdexcept>
#include <string>

namespace gvirtus::util {

  class SignalException : public std::runtime_error {
  public:
    SignalException(const std::string &message) : std::runtime_error(message) {}
  };
} // namespace gvirtus::util

#endif // GVIRTUS_SIGEXCEPT_H