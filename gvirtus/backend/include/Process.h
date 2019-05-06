#ifndef GVIRTUS_PROCESS_H
#define GVIRTUS_PROCESS_H

#include "communicator/Communicator.h"
#include "Handler.h"
#include "util/Observable.h"
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include <memory>
#include <string>
#include <vector>

namespace gvirtus {

  /**
   * Process is the object used by the Backend to handle the request from a single
   * Frontend.
   */
  class Process : public Observable {
  public:
    Process(const std::unique_ptr<comm::Communicator> communicator, std::vector<std::string> &plugins);
    ~Process() = default;
    void Execute(std::unique_ptr<comm::Communicator> client_comm);
    void Start();

  private:
    std::unique_ptr<comm::Communicator> _server_communicator;
    std::vector<std::string> mPlugins;
    std::vector<Handler *> mHandlers;
    log4cplus::Logger logger;
  };

} // namespace gvirtus
#endif /* GVIRTUS_PROCESS_H */