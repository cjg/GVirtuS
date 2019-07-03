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
#include <tuple>
#include "util/LD_Lib.h"

namespace gvirtus {

  /**
   * Process is the object used by the Backend to handle the request from a single
   * Frontend.
   */
  class Process : public Observable {
  public:
    Process(std::shared_ptr<LD_Lib<Communicator, std::string>> communicator, std::vector<std::string> &plugins);
    ~Process();
    void Execute(std::shared_ptr<Communicator> client_comm);
    void Start();

  private:
    std::shared_ptr<LD_Lib<Communicator, std::string>> _communicator;
    std::vector<std::shared_ptr<LD_Lib<Handler>>> _handlers;

    std::vector<std::string> mPlugins;
    log4cplus::Logger logger;
  };

} // namespace gvirtus
#endif /* GVIRTUS_PROCESS_H */