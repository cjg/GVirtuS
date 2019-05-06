#include <UvwAdapter.h>

// for sleep()
//#include <unistd.h>

namespace gvirtus::comm {
  //    UvwAdapter::UvwAdapter() : Communicator() {
  //        _loop = uvw::Loop::getDefault();
  //        _tcp = _loop->resource<uvw::TCPHandle>();
  //    }

  void
  UvwAdapter::Serve() {
    _tcp->once<uvw::ListenEvent>([](const uvw::ListenEvent &, uvw::TCPHandle &srv) {
      auto client = srv.loop().resource<uvw::TCPHandle>();

      client->on<uvw::CloseEvent>([ptr = srv.shared_from_this()](const uvw::CloseEvent &, uvw::TCPHandle &) { ptr->close(); });

      client->once<uvw::EndEvent>([](const uvw::EndEvent &, uvw::TCPHandle &client) { client.close(); });

      client->on<uvw::DataEvent>([](const uvw::DataEvent &, uvw::TCPHandle &) { /* data received */ });

      srv.accept(*client);
      client->read();
    });

    //        _tcp->bind();
    _tcp->listen();
    //      sleep(5);
  }

  void
  UvwAdapter::run() {
    Serve();
    //        _loop->run<uvw::Loop::Mode::ONCE>();
    //      _loop->run<uvw::Loop::Mode::NOWAIT>();
  }
} // namespace gvirtus::comm