#ifndef GVIRTUSPP_PROPERTY_H
#define GVIRTUSPP_PROPERTY_H

#include <iostream>
#include <string>
#include <vector>
#include <communicator/endpoint/Endpoint.h>

#include <nlohmann/json.hpp>

namespace gvirtus {
    /**
     * Property class.
     * This class is a model to represent the properties of the backend.
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
        Property(const int *endpoints,
                 const std::vector<std::string> *plugins);

        /**
         * This method is a setter for the class member _endpoints
         * @param endpoints: pointer to a vector of Endpoint
         * @return reference to itself (Fluent Interface API)
         */
        Property &endpoints(const int *endpoints);

        /**
         * This method is a getter for the class member _endpoints
         * @return the reference to vector where Endpoint are saved
         */
        inline const int &endpoints() const {
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
        }

    private:
        std::vector<std::string> _plugins;
        int _endpoints;
    };

    /**
     * This function will be used by nlohmann::json object when we call j.get&lt;Property&lt;().
     * Without this function the program doesn't know how to build a property object from json object.
     * @param j: reference to json object which contains the data
     * @param p: reference to property object to be created
     */
    inline void from_json(const nlohmann::json &j, Property &p) {
        int endpoints = 0;
        for (auto &el : j["endpoint"])
            endpoints++;

        std::vector<std::string> plugins;
        for (auto &el : j["plugins"])
            plugins.emplace_back(el);

        p.endpoints(&endpoints);
        p.plugins(&plugins);
    }
}

#endif //GVIRTUSPP_PROPERTY_H
