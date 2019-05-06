#ifndef GVIRTUS_COMMUNICATORFACTORY_H
#define GVIRTUS_COMMUNICATORFACTORY_H

#include "Communicator.h"
#include "Endpoint.h"
// FIXME: Endpoint_Tcp.h -> non deve essere qui, castare altrove.
#include "Endpoint_Tcp.h"
#include "TcpCommunicator.h"
#include <memory>

namespace gvirtus::comm {
  class CommunicatorFactory {
  public:
    static std::unique_ptr<Communicator>
    get_communicator(const std::shared_ptr<Endpoint> &end) {

      if (end->protocol() == "tcp") {
        std::shared_ptr<Endpoint_Tcp> end_tcp = std::dynamic_pointer_cast<Endpoint_Tcp>(end);
        return std::make_unique<TcpCommunicator>(end_tcp->address().c_str(), end_tcp->port());
      }

      return nullptr;
    }
  };
} // namespace gvirtus::comm

#endif // GVIRTUS_COMMUNICATORFACTORY_H