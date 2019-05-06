#ifndef GVIRTUS_ENDPOINTFACTORY_H
#define GVIRTUS_ENDPOINTFACTORY_H

#include "Endpoint.h"
#include "Endpoint_Tcp.h"
#include "util/JSON.h"
#include <filesystem>
#include <memory>
#include <nlohmann/json.hpp>

namespace gvirtus::comm {

  class EndpointFactory {
  public:
    static std::shared_ptr<Endpoint>
    get_endpoint(const std::filesystem::path &json_path) {
      std::shared_ptr<Endpoint> ptr;
      std::ifstream ifs(json_path);
      nlohmann::json j;
      ifs >> j;

      if ("tcp/ip" == j[ind_endpoint]["endpoint"].at("suite")) {
        //TODO: LOAD DYNAMIC LIBRARY COMM
        auto end = gvirtus::util::JSON<Endpoint_Tcp>(json_path).parser();
        ptr = std::make_shared<Endpoint_Tcp>(end);
      }

      ind_endpoint++;

      j.clear();
      ifs.close();

      return ptr;
    }

    static int
    index() {
      return ind_endpoint;
    }

  private:
    static int ind_endpoint;
  };
} // namespace gvirtus::comm

#endif // GVIRTUS_ENDPOINTFACTORY_H
