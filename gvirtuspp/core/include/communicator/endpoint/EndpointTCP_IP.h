#ifndef GVIRTUSPP_ENDPOINTTCP_IP_H
#define GVIRTUSPP_ENDPOINTTCP_IP_H

#include <communicator/endpoint/IEndpoint.h>
#include <communicator/endpoint/Endpoint.h>
#include <nlohmann/json.hpp>

namespace gvirtus {
    class EndpointTCP_IP : public IEndpoint {
    public:
        EndpointTCP_IP() = default;

        explicit EndpointTCP_IP(const std::string &endp_suite, const std::string &endp_protocol,
                                const std::string &endp_address, const std::string &endp_port);

        EndpointTCP_IP(const std::string &endp_suite) : EndpointTCP_IP(endp_suite,
                                                                       "tcp",
                                                                       "127.0.0.1",
                                                                       "9999") {}

        IEndpoint &suite(const std::string &suite) override;

        IEndpoint &protocol(const std::string &protocol) override;

        /**
         * This method is a setter for the class member _address
         * @param address: string containing address value
         * @return reference to itself (Fluent Interface API)
         */
        EndpointTCP_IP &address(const std::string &address);

        /**
         * This method is a setter for the class member _port
         * @param port: string containing port value
         * @return reference to itself (Fluent Interface API)
         */
        EndpointTCP_IP &port(const std::string &port);

        /**
           * This method is a getter for the class member _address
           * @return reference to class member _address
           */
        inline const std::string &address() const {
            return _address;
        }

        /**
         * This method is a getter for the class member _port
         * @return reference to class member _port
         */
        inline const std::uint16_t &port() const {
            return _port;
        }

        /**
         * This method return an object description
         * @return string that represents the concatenation between class member
         */
        virtual inline const std::string to_string() const {
            return _suite + _protocol + _address + std::to_string(_port);
        };

    private:
        std::string _address;
        std::uint16_t _port;
    };

    /**
     * This function will be used by nlohmann::json object when we call j.get&lt;Property&lt;().
     * Without this function the program doesn't know how to build a property object from json object.
     * @param j: reference to json object which contains the data
     * @param p: reference to endpoint object to be created
     */
    inline void from_json(const nlohmann::json &j, EndpointTCP_IP &end) {
        auto el = j["endpoint"][Endpoint::index()];

        end.suite(el.at("suite"));
        end.protocol(el.at("protocol"));
        end.address(el.at("address"));
        end.port(el.at("port"));
    }
}

#endif //GVIRTUSPP_ENDPOINTTCP_IP_H
