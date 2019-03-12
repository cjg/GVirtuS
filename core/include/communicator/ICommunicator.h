#ifndef GVIRTUSPP_ICOMMUNICATOR_H
#define GVIRTUSPP_ICOMMUNICATOR_H

#include <communicator/endpoint/Endpoint.h>

namespace gvirtus {
    class ICommunicator {
    public:
        explicit ICommunicator(const Endpoint &endpoint) {
            _end = endpoint;
        };

        virtual void run() = 0;

        virtual void Serve() = 0;

        virtual void Accept() = 0;

        virtual void Connect() = 0;

        virtual std::size_t Read() = 0;

        virtual std::size_t Write() = 0;

        virtual void Close() = 0;

        virtual void Sync() = 0;


        virtual ~ICommunicator() {};
    protected:
        Endpoint _end;
    };
}
#endif //GVIRTUSPP_ICOMMUNICATOR_H
