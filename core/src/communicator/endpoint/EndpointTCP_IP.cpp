#include <communicator/endpoint/EndpointTCP_IP.h>
#include <regex>

namespace gvirtus {
    EndpointTCP_IP::EndpointTCP_IP(const std::string &endp_suite, const std::string &endp_protocol,
                                   const std::string &endp_address, const std::string &endp_port) {
        suite(endp_suite);
        protocol(endp_protocol);
        address(endp_address);
        port(endp_port);
    }

    IEndpoint &EndpointTCP_IP::suite(const std::string &suite) {
        std::regex pattern{R"([[:alpha:]]*/[[:alpha:]]*)"};

        std::smatch matches;

        std::regex_search(suite, matches, pattern);

        if (suite == matches[0])
            _suite = suite;

        return *this;
    }

    IEndpoint &EndpointTCP_IP::protocol(const std::string &protocol) {
        std::regex pattern{R"([[:alpha:]]*)"};

        std::smatch matches;

        std::regex_search(protocol, matches, pattern);

        if (protocol == matches[0])
            _protocol = protocol;

        return *this;
    }

    EndpointTCP_IP &EndpointTCP_IP::address(const std::string &address) {
        std::regex pattern{
                R"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)"};
        std::smatch matches;

        std::regex_search(address, matches, pattern);

        if (address == matches[0])
            _address = address;

        return *this;
    }

    EndpointTCP_IP &EndpointTCP_IP::port(const std::string &port) {
        std::regex pattern{
                R"((6553[0-5]|655[0-2][0-9]\d|65[0-4](\d){2}|6[0-4](\d){3}|[1-5](\d){4}|[1-9](\d){0,3}))"};

        std::smatch matches;

        std::regex_search(port, matches, pattern);

        if (port == matches[0])
            _port = (uint16_t) std::stoi(port);

        return *this;
    }
}