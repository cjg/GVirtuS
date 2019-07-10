#ifndef GVIRTUS_COMMUNICATORFACTORY_H
#define GVIRTUS_COMMUNICATORFACTORY_H

#include "Communicator.h"
#include "Endpoint.h"
#include "Endpoint_Tcp.h"
#include "util/LD_Lib.h"
#include <memory>
#include <utility>
#include <vector>

#include <iostream>
namespace gvirtus {

class CommunicatorFactory {
 public:
  static std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<Endpoint>>>
  get_communicator(std::shared_ptr<Endpoint> end, bool secure = false) {
    std::shared_ptr<LD_Lib<Communicator, std::shared_ptr<Endpoint>>> dl;

    if (!secure) {
      if (end->protocol() == "tcp" || end->protocol() == "http" || end->protocol() == "oldtcp"
          || end->protocol() == "ws") { //an array with supported communicators is better and elegant
        dl = std::make_shared<LD_Lib<Communicator, std::shared_ptr<Endpoint>>>(_COMMS_DIR
        "/lib" + end->protocol() + "-communicator.so", "create_communicator");
      } else
        throw std::runtime_error("Unsecure communicator not supported");
    } else if (end->protocol() == "https" || end->protocol() == "wss") { //in secure supported
      dl = std::make_shared<LD_Lib<Communicator, std::shared_ptr<Endpoint>>>(_COMMS_DIR
      "/lib" + end->protocol() + "-communicator.so", "create_communicator");
    } else {
      throw std::runtime_error("Secure communicator not supported");
    }

    dl->build_obj(end);
    return dl;
  }
};
} // namespace gvirtus

#endif // GVIRTUS_COMMUNICATORFACTORY_HS_COMMUNICATORFACTORY_H