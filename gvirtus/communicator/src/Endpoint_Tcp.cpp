#include "Endpoint_Tcp.h"
#include "EndpointFactory.h"
#include <regex>

namespace gvirtus {
  Endpoint_Tcp::Endpoint_Tcp(const std::string &endp_suite, const std::string &endp_protocol, const std::string &endp_address, const std::string &endp_port) {
    suite(endp_suite);
    protocol(endp_protocol);
    address(endp_address);
    port(endp_port);
  }

  Endpoint &
  Endpoint_Tcp::suite(const std::string &suite) {
    std::regex pattern{R"([[:alpha:]]*/[[:alpha:]]*)"};

    std::smatch matches;

    std::regex_search(suite, matches, pattern);

    if (suite == matches[0])
      _suite = suite;

    return *this;
  }

  Endpoint &
  Endpoint_Tcp::protocol(const std::string &protocol) {
    std::regex pattern{R"([[:alpha:]]*)"};

    std::smatch matches;

    std::regex_search(protocol, matches, pattern);

    if (protocol == matches[0])
      _protocol = protocol;

    return *this;
  }

  Endpoint_Tcp &
  Endpoint_Tcp::address(const std::string &address) {
    std::regex pattern{
        R"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)"};
    std::smatch matches;

    std::regex_search(address, matches, pattern);

    if (address == matches[0])
      _address = address;

    return *this;
  }

  Endpoint_Tcp &
  Endpoint_Tcp::port(const std::string &port) {
    std::regex pattern{
        R"((6553[0-5]|655[0-2][0-9]\d|65[0-4](\d){2}|6[0-4](\d){3}|[1-5](\d){4}|[1-9](\d){0,3}))"};

    std::smatch matches;

    std::regex_search(port, matches, pattern);

    if (port == matches[0])
      _port = (uint16_t)std::stoi(port);

    return *this;
  }

  void
  from_json(const nlohmann::json &j, Endpoint_Tcp &end) {
    auto el = j["communicator"][EndpointFactory::index()]["endpoint"];

    end.suite(el.at("suite"));
    end.protocol(el.at("protocol"));
    end.address(el.at("server_address"));
    end.port(el.at("port"));
  }

} // namespace gvirtus