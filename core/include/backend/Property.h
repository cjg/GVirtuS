#ifndef GVIRTUSPP_PROPERTY_H
#define GVIRTUSPP_PROPERTY_H

#include <iostream>
#include <string>
#include <vector>
#include <backend/Endpoint.h>

#include <nlohmann/json.hpp>

namespace gvirtus {
    /**
     *
     */
    class Property {
    public:
        /**
         * Default constructor
         */
        Property() = default;

        /**
         * Parameterized constructor
         * @param endpoints: pointer to a vector of Endpoint
         * @param plugins: pointer to a vector of std::string
         */
        Property(const std::vector<gvirtus::Endpoint> *endpoints,
                 const std::vector<std::string> *plugins);

        /**
         * This method is a setter for the class member _endpoints
         * @param endpoints: pointer to a vector of Endpoint
         * @return reference to itself (Fluent Interface API)
         */
        Property &endpoints(const std::vector<gvirtus::Endpoint> *endpoints);

        /**
         * This method is a getter for the class member _endpoints
         * @return the reference to vector where Endpoint are saved
         */
        inline const std::vector<gvirtus::Endpoint> &endpoints() const {
            return _endpoints;
        }

        /**
         * This method is a setter for the class member _plugins
         * @param plugins: pointer to a vector of std::string
         * @return reference to itself (Fluent Interface API)
         */
        Property &plugins(const std::vector<std::string> *plugins);

        /**
         * This method is a getter for the class member _plugins
         * @return the reference to vector where plugins string are saved
         */
        inline const std::vector<std::string> &plugins() const {
            return _plugins;
        }

        /**
         * Class destroyer
         */
        ~Property() {
            _plugins.clear();
            _endpoints.clear();
        }

    private:
        std::vector<std::string> _plugins;
        std::vector<gvirtus::Endpoint> _endpoints;
    };

    /**
     * This function will be used by nlohmann::json object when we call j.get&lt;Property&lt;().
     * Without this function the program doesn't know how to build a property object from json object.
     * @param j: reference to json object which contains the data
     * @param p: reference to property object to be created
     */
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
