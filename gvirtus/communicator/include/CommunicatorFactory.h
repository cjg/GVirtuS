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
  static std::shared_ptr<LD_Lib<Communicator, std::string>>
  get_communicator(const std::shared_ptr<Endpoint> &end, bool secure = false) {
    if (!secure) {
      if (end->protocol() == "tcp") {
        std::shared_ptr<Endpoint_Tcp> end_tcp = std::dynamic_pointer_cast<Endpoint_Tcp>(end);
        auto dl = std::make_shared<LD_Lib<Communicator, std::string>>(_COMMS_DIR "/libtcp-communicator.so", "create_communicator");
        dl->build_obj(end_tcp->address() + ":" + std::to_string(end_tcp->port()));
        return dl;
      } else if (end->protocol() == "http") {
        std::shared_ptr<Endpoint_Tcp> end_tcp = std::dynamic_pointer_cast<Endpoint_Tcp>(end);
        auto dl = std::make_shared<LD_Lib<Communicator, std::string>>(_COMMS_DIR "/libhttp-communicator.so", "create_communicator");
        dl->build_obj(end_tcp->address() + ":" + std::to_string(end_tcp->port()));
        return dl;
      }
    } else {
      if (end->protocol() == "https") {
        std::shared_ptr<Endpoint_Tcp> end_tcp = std::dynamic_pointer_cast<Endpoint_Tcp>(end);
        std::cout << "todo HTTPS" << std::endl;
      }
    }

    throw std::string("Communicator not supported");
  }
};
} // namespace gvirtus

#endif // GVIRTUS_COMMUNICATORFACTORY_H