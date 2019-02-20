#include <backend/Endpoint.h>
#include <regex>
#include <exception/EndpointException.h>

namespace gvirtus {
    Endpoint::Endpoint(const std::string &transmit_protocol, const std::string &transmit_address,
                       const std::string &transmit_port) {
        protocol(transmit_protocol);
        address(transmit_address);
        port(transmit_port);
    }

    Endpoint &Endpoint::protocol(const std::string &protocol) {
        std::regex pattern{R"([[:alpha:]]*)"};
        std::smatch matches;

        std::regex_search(protocol, matches, pattern);

        if (protocol != matches[0])
            throw EndpointException("Property.cpp", 20, "protocol(const std::string &protocol)",
                                    "Invalid protocol.");
        else
            _protocol = protocol;

        return *this;
    }

    Endpoint &Endpoint::address(const std::string &address) {
        std::regex pattern{
                R"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)"};
        std::smatch matches;

        std::regex_search(address, matches, pattern);

        if (address != matches[0])
            throw EndpointException("Property.cpp", 36, "address(const std::string &address)", "Invalid address.");
        else
            _address = address;

        return *this;
    }

    Endpoint &Endpoint::port(const std::string &port) {
        std::regex pattern{R"((6553[0-5]|655[0-2][0-9]\d|65[0-4](\d){2}|6[0-4](\d){3}|[1-5](\d){4}|[1-9](\d){0,3}))"};

        std::smatch matches;

        std::regex_search(port, matches, pattern);

        if (port != matches[0])
            throw EndpointException("Property.cpp", 51, "port(const std::string &port)", "Invalid port number.");
        else
            _port = (uint16_t) std::stoi(port);

        return *this;
    }
}