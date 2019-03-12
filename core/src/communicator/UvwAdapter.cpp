#include <communicator/UvwAdapter.h>

//for sleep()
//#include <unistd.h>

namespace gvirtus {
    UvwAdapter::UvwAdapter(const Endpoint &end) : ICommunicator(end) {
        _loop = uvw::Loop::getDefault();
        _tcp = _loop->resource<uvw::TCPHandle>();
        _tcp_end = std::dynamic_pointer_cast<gvirtus::EndpointTCP_IP>(end.get());
    }

    void UvwAdapter::Serve() {
        _tcp->once<uvw::ListenEvent>([](const uvw::ListenEvent &, uvw::TCPHandle &srv) {
            auto client = srv.loop().resource<uvw::TCPHandle>();

            client->on<uvw::CloseEvent>(
                    [ptr = srv.shared_from_this()](const uvw::CloseEvent &, uvw::TCPHandle &) { ptr->close(); });
            client->once<uvw::EndEvent>([](const uvw::EndEvent &, uvw::TCPHandle &client) { client.close(); });
            client->on<uvw::DataEvent>([](const uvw::DataEvent &, uvw::TCPHandle &) {/* data received */});
            srv.accept(*client);
            client->read();
        });

        _tcp->bind(_tcp_end->address(), _tcp_end->port());
        _tcp->listen();
//      sleep(5);
    }

    void UvwAdapter::run() {
        Serve();
        _loop->run<uvw::Loop::Mode::ONCE>();
//      _loop->run<uvw::Loop::Mode::NOWAIT>();
    }
}