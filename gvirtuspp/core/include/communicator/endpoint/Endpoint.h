#ifndef GVIRTUSPP_ENDPOINT_H
#define GVIRTUSPP_ENDPOINT_H

#include <communicator/endpoint/IEndpoint.h>
#include <memory>
#include <filesystem>

namespace gvirtus {
    class Endpoint {
    public:
        Endpoint() = default;

        explicit Endpoint(const std::filesystem::path &_json_path);

        std::shared_ptr<IEndpoint> get() const {
            return _ptr_i_endpoint;
        }

        ~Endpoint();

        static int index() {
            return ind_endpoint;
        }
    private:
        static int ind_endpoint;
        std::shared_ptr<IEndpoint> _ptr_i_endpoint;
    };
}
#endif //GVIRTUSPP_ENDPOINT_H
