#pragma once

#include <gvirtus/common/LD_Lib.h>
#include <gvirtus/common/Observable.h>
#include <gvirtus/communicators/Communicator.h>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include "Handler.h"
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

namespace gvirtus::backend {
/**
 * Process is the object used by the Backend to handle the request from a single
 * Frontend.
 */
class Process : public common::Observable {
 public:
  Process(
      std::shared_ptr<common::LD_Lib<communicators::Communicator,
                                     std::shared_ptr<communicators::Endpoint>>>
          communicator,
      std::vector<std::string> &plugins);
  ~Process() override;
  void Start();

 private:
  std::shared_ptr<common::LD_Lib<communicators::Communicator,
                                 std::shared_ptr<communicators::Endpoint>>>
      _communicator;
  std::vector<std::shared_ptr<common::LD_Lib<Handler>>> _handlers;

  std::vector<std::string> mPlugins;
  log4cplus::Logger logger;
};
}  // namespace gvirtus::backend
