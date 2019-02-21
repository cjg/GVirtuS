#ifndef GVIRTUSPP_ENDPOINT_H
#define GVIRTUSPP_ENDPOINT_H

#include <string>

namespace gvirtus {
    /**
     * Endpoint class.
     * This class is a model to represent the endpoint property.
     */
    class Endpoint {
    public:
        /**
         * Default constructor
         */
        Endpoint() = default;

        /**
         * Parameterized constructor
         * @param protocol: string containing protocol value
         * @param address: string containing address value
         * @param port: string containing port value
         */
        Endpoint(const std::string &protocol, const std::string &address, const std::string &port);

        /**
         * This method is a setter for the class member _protocol
         * @param protocol: string containing protocol value
         * @return reference to itself (Fluent Interface API)
         */
        Endpoint &protocol(const std::string &protocol);

        /**
         * This method is a setter for the class member _address
         * @param address: string containing address value
         * @return reference to itself (Fluent Interface API)
         */
        Endpoint &address(const std::string &address);

        /**
         * This method is a setter for the class member _port
         * @param port: string containing port value
         * @return reference to itself (Fluent Interface API)
         */
        Endpoint &port(const std::string &port);

        /**
         * This method is a getter for the class member _protocol
         * @return reference to class member _protocol
         */
        inline const std::string &protocol() const {
            return _protocol;
        }

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

    private:
        std::string _protocol;
        std::string _address;
        std::uint16_t _port = 9999;
    };
}
#endif //GVIRTUSPP_ENDPOINT_H
