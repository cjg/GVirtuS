#ifndef GVIRTUSPP_PROPERTY_H
#define GVIRTUSPP_PROPERTY_H

#include <iostream>
#include <string>
#include <vector>
#include <backend/Endpoint.h>

#include <nlohmann/json.hpp>

namespace gvirtus {
    class Property {
    public:
        Property() = default;

        Property(const std::vector<gvirtus::Endpoint> *endpoints,
                 const std::vector<std::string> *plugins);

        Property &endpoints(const std::vector<gvirtus::Endpoint> *endpoints);

        inline const std::vector<gvirtus::Endpoint> &endpoints() const {
            return _endpoints;
        }

        Property &plugins(const std::vector<std::string> *plugins);

        inline const std::vector<std::string> &plugins() const {
            return _plugins;
        }

        ~Property() {
            _plugins.clear();
            _endpoints.clear();
        }

    private:
        std::vector<std::string> _plugins;
        std::vector<gvirtus::Endpoint> _endpoints;
    };

    inline void from_json(const nlohmann::json &j, Property &p) {
        std::vector<gvirtus::Endpoint> endpoints;

        for (auto &el : j["endpoint"]) {
            gvirtus::Endpoint epoint;
            epoint.protocol(el.at("protocol")).address(el.at("address")).port(el.at("port"));
            endpoints.emplace_back(epoint);
        }

        p.endpoints(&endpoints);

        std::vector<std::string> plugins;
        for (auto &el : j["plugins"])
            plugins.emplace_back(el);

        p.plugins(&plugins);
    }
}

#endif //GVIRTUSPP_PROPERTY_H
