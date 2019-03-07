#include <communicator/endpoint/Endpoint.h>
#include <communicator/endpoint/EndpointTCP_IP.h>

#include <iostream>
#include <util/JSON.h>


namespace gvirtus {
    int Endpoint::ind_endpoint = 0;

    Endpoint::Endpoint(const std::filesystem::path &_json_path) {
        std::ifstream ifs(_json_path);
        nlohmann::json j;
        ifs >> j;

        if ("tcp/ip" == j["endpoint"][ind_endpoint].at("suite")) {
            EndpointTCP_IP tcp_ip = gvirtus::util::JSON<EndpointTCP_IP>(_json_path).parser();
            _ptr_i_endpoint = std::make_shared<EndpointTCP_IP>(tcp_ip);

        }
        //Add here: Else if new suite {}

        ind_endpoint++;
        j.clear();
        ifs.close();
    }

    Endpoint::~Endpoint() {
        _ptr_i_endpoint.reset();
    }
}