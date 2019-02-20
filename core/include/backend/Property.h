#ifndef GVIRTUSPP_PROPERTY_H
#define GVIRTUSPP_PROPERTY_H

#include <string>
#include <vector>
#include <backend/Endpoint.h>

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

    private:
        std::vector<std::string> _plugins;
        std::vector<gvirtus::Endpoint> _endpoints;
    };
}

#endif //GVIRTUSPP_PROPERTY_H
