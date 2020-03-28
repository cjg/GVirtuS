#pragma once

#include <gvirtus/common/Observer.h>

#include <string>
#include <thread>
#include <vector>
#include "Process.h"
#include "Property.h"
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

namespace gvirtus::backend {
/**
 * Backend is the main object of gvirtus-backend. It is responsible of accepting
 * the connection from the Frontend(s) and spawing a new Process for handling
 * each Frontend.
 */
class Backend : public common::Observer {
 public:
  Backend(const fs::path &path);
  /**
   * Starts the Backend. The call to Start() will make the Backend to serve
   * forever.
   */
  void Start();
  void EventOccurred(std::string &event, void *object);
  virtual ~Backend() = default;

 private:
  std::vector<std::unique_ptr<Process>> _children;
  Property _properties;
  log4cplus::Logger logger;
};
}  // namespace gvirtus::backend