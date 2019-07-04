/**
 * @mainpage gVirtuS - A GPGPU transparent virtualization component
 *
 * @section Introduction
 * gVirtuS tries to fill the gap between in-house hosted computing clusters,
 * equipped with GPGPUs devices, and pay-for-use high performance virtual
 * clusters deployed  via public or private computing clouds. gVirtuS allows an
 * instanced virtual machine to access GPGPUs in a transparent way, with an
 * overhead  slightly greater than a real machine/GPGPU setup. gVirtuS is
 * hypervisor independent, and, even though it currently virtualizes nVIDIA CUDA
 * based GPUs, it is not limited to a specific brand technology. The performance
 * of the components of gVirtuS is assessed through a suite of tests in
 * different deployment scenarios, such as providing GPGPU power to cloud
 * computing based HPC clusters and sharing remotely hosted GPGPUs among HPC
 * nodes.
 */

#include "Backend.h"
#include "Property.h"
#include "communicator/Communicator.h"
#include "communicator/CommunicatorFactory.h"
#include "communicator/EndpointFactory.h"
#include "util/JSON.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <unistd.h>
#include <stdlib.h> /* getenv */
#include <string>

#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"

// test
#include "util/LD_Lib.h"

log4cplus::Logger logger;

int
main(int argc, char **argv) {
  //Logger configuration
  log4cplus::BasicConfigurator config;
  config.configure();
  logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("GVirtuS"));
  LOG4CPLUS_INFO(logger, "🛈  - GVirtuS backend version");


  std::string config_path;
#ifdef _CONFIG_FILE_JSON
  config_path = _CONFIG_FILE_JSON;
#endif
  if (argc == 2) {
    config_path = std::string(argv[1]);
  }

  LOG4CPLUS_INFO(logger, "🛈  - Configuration: " << config_path);

  //FIXME: Try - Catch? No.
  try {
    gvirtus::Backend b(config_path);

    LOG4CPLUS_INFO(logger, "🛈  - Up and running");
    b.Start();

  } catch (std::string &e) {
    LOG4CPLUS_ERROR(logger, "✖ - Exception:" << e);
  } catch (const char *e) {
    LOG4CPLUS_ERROR(logger, "✖ - Exception:" << e);
  }

  LOG4CPLUS_INFO(logger, "🛈  - [Process " << getpid() << "] Shutdown");
  return 0;
}