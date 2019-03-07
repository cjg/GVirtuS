#ifndef GVIRTUSPP_IENDPOINT_H
#define GVIRTUSPP_IENDPOINT_H

#include <string>

namespace gvirtus {
    /**
     * Endpoint class.
     * This class is a model to represent the endpoint property.
     */
    class IEndpoint {
    public:
        /**
         * Parameterized constructor
         * @param protocol: string containing protocol value
         * @param address: string containing address value
         * @param port: string containing port value
         */
        IEndpoint() = default;

        /**
         * This method is a setter for the class member _suite
         * @param protocol: string containing suite value
         * @return reference to itself (Fluent Interface API)
         */
        virtual IEndpoint &suite(const std::string &suite) = 0;

        /**
         * This method is a getter for the class member _suite
         * @return reference to class member _suite
         */
        virtual inline const std::string &suite() const {
            return _suite;
        }

        /**
         * This method is a setter for the class member _protocol
         * @param protocol: string containing protocol value
         * @return reference to itself (Fluent Interface API)
         */
        virtual IEndpoint &protocol(const std::string &protocol) = 0;

        /**
         * This method is a getter for the class member _protocol
         * @return reference to class member _protocol
         */
        virtual inline const std::string &protocol() const {
            return _protocol;
        }

        /**
         * This method return an object description
         * @return string that represents the concatenation between class member
         */
        virtual inline const std::string to_string() const {
            return _suite + _protocol;
        };

        /**
         * This method is an overload of == operator.
         * @param endpoint: object to compare
         * @return True if two object are similar, false otherwise
         */
        inline bool operator==(const IEndpoint &endpoint) const {
            return this->to_string() == endpoint.to_string();
        }

    protected:
        std::string _suite;
        std::string _protocol;
    };
}

#endif //GVIRTUSPP_IENDPOINT_H
