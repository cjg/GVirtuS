#ifndef GVIRTUS_HTTPADAPTER_H
#define GVIRTUS_HTTPADAPTER_H

#include "Communicator.h"
#include <uWebSockets/App.h>
#include <thread>
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"


namespace gvirtus {
class HTTPAdapter : public Communicator {
 public:
  HTTPAdapter() = default;
  explicit HTTPAdapter(const std::string &communicator);
  HTTPAdapter(std::string &hostname, std::string &port);
  virtual ~HTTPAdapter();

  void run() override;

  void Serve() override;

  const Communicator *const Accept() const override;

  void Connect() override;

  std::size_t Read(char *buffer, size_t size) override {};

  std::size_t Write(const char *buffer, size_t size) override {};

  void Close() override {};

  void Sync() override {};

 private:
  std::vector<std::unique_ptr<std::thread>> _threads;
  std::string _hostname;
  std::string _port;
  log4cplus::Logger logger;
};
} // namespace gvirtus
#endif // GVIRTUS_HTTPADAPTER_H
