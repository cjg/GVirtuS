#include <backend/Property.h>
#include <regex>
#include <iostream>

Property::Property(const std::string &transmit_protocol, const std::string &transmit_address,
                   const std::string &transmit_port,
                   const std::vector<std::string> *plugins_required) {
    protocol(transmit_protocol);
    address(transmit_address);
    port(transmit_port);
    plugins(plugins_required);
}

const std::string &Property::protocol() const {
    return _protocol;
}

const std::string &Property::address() const {
    return _address;
}

const std::uint16_t &Property::port() const {
    return _port;
}

const std::vector<std::string> &Property::plugins() const {
    return _plugins;
}

Property &Property::protocol(const std::string &protocol) {
    std::regex pattern{R"([[:alpha:]]*)"};
    std::smatch matches;

    std::regex_search(protocol, matches, pattern);

    if (protocol != matches[0]) {
        //TODO: launch exception (REGEX EXCEPTION)
        ;
    } else
        _protocol = protocol;

    return *this;
}

Property &Property::address(const std::string &address) {
    std::regex pattern{
            R"(^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$)"};
    std::smatch matches;

    std::regex_search(address, matches, pattern);

    if (address != matches[0]) {
        //TODO: launch exception (REGEX EXCEPTION)
        ;
    } else
        _address = address;

    return *this;
}

Property &Property::port(const std::string &port) {
    std::regex pattern{R"((6553[0-5]|655[0-2][0-9]\d|65[0-4](\d){2}|6[0-4](\d){3}|[1-5](\d){4}|[1-9](\d){0,3}))"};

    std::smatch matches;

    std::regex_search(port, matches, pattern);

    if (port != matches[0]) {
        //TODO: launch exception (REGEX EXCEPTION)
        ;
    } else
        _port = (uint16_t) std::stoi(port);

    return *this;
}

Property &Property::plugins(const std::vector<std::string> *plugins) {
    if (plugins == nullptr || plugins->empty()) {
        //TODO: fill plugins with all plugin
        _plugins.emplace_back("nullptr, enable all plugin");
    } else
        _plugins = *plugins;

    return *this;
}