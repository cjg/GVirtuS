#include <communicator/Communicator.h>
#include <communicator/UvwAdapter.h>
#include <communicator/endpoint/Endpoint.h>

namespace gvirtus {
    Communicator::Communicator(const gvirtus::Endpoint &end) {
        if (end.get().get()->protocol() == "tcp")
           _ptr_i_communicator = std::make_shared<UvwAdapter>(end);
    }

    Communicator::~Communicator() {
        _ptr_i_communicator.reset();
    }
}
