#pragma once

#include <gvirtus/common/JSON.h>
#include <memory>
#include <nlohmann/json.hpp>
#include "Endpoint.h"
#include "Endpoint_Tcp.h"

namespace gvirtus::communicators {
class EndpointFactory {
 public:
  static std::shared_ptr<Endpoint> get_endpoint(const fs::path &json_path) {
    std::shared_ptr<Endpoint> ptr;
    std::ifstream ifs(json_path);
    nlohmann::json j;
    ifs >> j;

    if ("tcp/ip" == j["communicator"][ind_endpoint]["endpoint"].at("suite")) {
      auto end = common::JSON<Endpoint_Tcp>(json_path).parser();
      ptr = std::make_shared<Endpoint_Tcp>(end);
    }

    ind_endpoint++;

    j.clear();
    ifs.close();

    return ptr;
  }

  static int index() { return ind_endpoint; }

 private:
  static int ind_endpoint;
};
}  // namespace gvirtus::communicators
