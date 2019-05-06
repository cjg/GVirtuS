#ifndef GVIRTUS_BACKEND_H
#define GVIRTUS_BACKEND_H

#include "util/Observer.h"
#include "Property.h"
#include "Process.h"
#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace gvirtus {

  /**
   * Backend is the main object of gvirtus-backend. It is responsible of accepting
   * the connection from the Frontend(s) and spawing a new Process for handling
   * each Frontend.
   */
  class Backend : public Observer {
  public:
    Backend(const std::filesystem::path &path);
    /**
     * Starts the Backend. The call to Start() will make the Backend to serve
     * forever.
     */
    void Start();
    void EventOccurred(std::string &event, void *object);
    virtual ~Backend() = default;

  private:
    std::vector<std::unique_ptr<gvirtus::Process>> _children;
    gvirtus::Property _properties;
    log4cplus::Logger logger;
  };

} // namespace gvirtus

#endif /* GVIRTUS_BACKEND_H */