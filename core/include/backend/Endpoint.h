#ifndef GVIRTUSPP_ENDPOINT_H
#define GVIRTUSPP_ENDPOINT_H

#include <string>
#include <vector>

namespace gvirtus {
    class Endpoint {
    public:
        Endpoint() = default;

        Endpoint(const std::string &protocol, const std::string &address, const std::string &port);

        Endpoint &protocol(const std::string &protocol);

        Endpoint &address(const std::string &address);

        Endpoint &port(const std::string &port);

        inline const std::string &protocol() const {
            return _protocol;
        }

        inline const std::string &address() const {
            return _address;
        }

        inline const std::uint16_t &port() const {
            return _port;
        }

    private:
        std::string _protocol;
        std::string _address;
        std::uint16_t _port = 9999;
    };
}
#endif //GVIRTUSPP_ENDPOINT_H
