#pragma once

#include <gvirtus/common/LD_Lib.h>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include "Communicator.h"
#include "Endpoint.h"
#include "Endpoint_Tcp.h"

namespace gvirtus::communicators {
class CommunicatorFactory {
 public:
  static std::shared_ptr<
      common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>>
  get_communicator(std::shared_ptr<Endpoint> end, bool secure = false) {
    std::shared_ptr<common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>> dl;

    if (!secure) {
      if (end->protocol() == "tcp" || end->protocol() == "http" ||
          end->protocol() == "oldtcp" ||
          end->protocol() == "ws") {  // an array with supported communicators
                                      // is better and elegant
        dl = std::make_shared<
            common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>>(
            std::string{GVIRTUS_HOME} + "/lib/libgvirtus-communicators-" +
                end->protocol() + ".so",
            "create_communicator");
      } else
        throw std::runtime_error("Unsecure communicator not supported");
    } else if (end->protocol() == "https" ||
               end->protocol() == "wss") {  // in secure supported
      dl = std::make_shared<
          common::LD_Lib<Communicator, std::shared_ptr<Endpoint>>>(
          std::string{GVIRTUS_HOME} + "/lib/libgvirtus-communicators-" +
              end->protocol() + ".so",
          "create_communicator");
    } else {
      throw std::runtime_error("Secure communicator not supported");
    }

    dl->build_obj(end);
    return dl;
  }
};
}  // namespace gvirtus::communicators
