#ifndef GVIRTUSPP_COMMUNICATOR_H
#define GVIRTUSPP_COMMUNICATOR_H

#include <communicator/ICommunicator.h>
#include <memory>

namespace gvirtus {
    class Communicator {
    public:
        explicit Communicator(const Endpoint &);

        ~Communicator();

        void run() {
            _ptr_i_communicator->run();
        }

    private:
        std::shared_ptr<ICommunicator> _ptr_i_communicator;
    };
}
#endif //GVIRTUSPP_COMMUNICATOR_H
