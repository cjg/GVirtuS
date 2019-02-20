#ifndef GVIRTUSPP_PROPERTY_H
#define GVIRTUSPP_PROPERTY_H

#include <string>
#include <vector>

class Property {
public:
    Property() = default;

    Property(const std::string &protocol, const std::string &address, const std::string &port,
             const std::vector<std::string> *plugins);

    const std::string &protocol() const;

    const std::string &address() const;

    const std::uint16_t &port() const;

    const std::vector<std::string> &plugins() const;

    Property &protocol(const std::string &protocol);

    Property &address(const std::string &address);

    Property &port(const std::string &port);

    Property &plugins(const std::vector<std::string> *plugins);

private:
    std::string _protocol;
    std::string _address;
    std::uint16_t _port = 65535;
    std::vector<std::string> _plugins;
};

#endif //GVIRTUSPP_PROPERTY_H
