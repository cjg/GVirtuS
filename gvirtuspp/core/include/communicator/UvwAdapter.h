#ifndef GVIRTUSPP_UVWADAPTER_H
#define GVIRTUSPP_UVWADAPTER_H

#include <uvw.hpp>
#include <communicator/ICommunicator.h>
#include <communicator/endpoint/Endpoint.h>
#include <communicator/endpoint/EndpointTCP_IP.h>

namespace gvirtus {
    class UvwAdapter : public ICommunicator {
    public:
        explicit UvwAdapter(const Endpoint &);

        virtual void run();

        virtual void Serve();

        virtual void Accept() {};

        virtual void Connect() {};

        virtual std::size_t Read() {};

        virtual std::size_t Write() {};

        virtual void Close() {};

        virtual void Sync() {};
    private:
        std::shared_ptr<uvw::Loop> _loop;
        std::shared_ptr<uvw::TCPHandle> _tcp;
        std::shared_ptr<gvirtus::EndpointTCP_IP> _tcp_end;
    };
}
#endif //GVIRTUSPP_UVWADAPTER_H
