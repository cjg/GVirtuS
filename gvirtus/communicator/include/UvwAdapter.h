#ifndef GVIRTUS_UVWADAPTER_H
#define GVIRTUS_UVWADAPTER_H

#include <Communicator.h>
#include <memory>
#include <uvw.hpp>

namespace gvirtus::comm {
  class UvwAdapter : public Communicator {
  public:
    explicit UvwAdapter(const std::string &communicator){};

    virtual void run();

    virtual void Serve();

    virtual void Accept(){};

    virtual void Connect(){};

    virtual std::size_t Read(){};

    virtual std::size_t Write(){};

    virtual void Close(){};

    virtual void Sync(){};

  private:
    std::shared_ptr<uvw::Loop> _loop;
    std::shared_ptr<uvw::TCPHandle> _tcp;
  };
} // namespace gvirtus::comm
#endif // GVIRTUS_UVWADAPTER_H
