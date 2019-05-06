#include "Backend.h"
#include "communicator/CommunicatorFactory.h"
#include "communicator/EndpointFactory.h"
#include "util/JSON.h"
#include <sys/wait.h>
#include <unistd.h>

namespace gvirtus {

  Backend::Backend(const std::filesystem::path &path) {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("Backend"));

    if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path) && path.extension() == ".json") {
      LOG4CPLUS_DEBUG(logger, "✓ - " << std::filesystem::path(__FILE__).filename() << ":" << __LINE__ << ":"
                                     << " Json file has been loaded.");

      _properties = gvirtus::util::JSON<Property>(path).parser();
      _children.reserve(_properties.endpoints());

      if (_properties.endpoints() > 1) {
        LOG4CPLUS_INFO(logger, "🛈  - Application serves on " << _properties.endpoints() << " several endpoint");
      }

      for (int i = 0; i < _properties.endpoints(); i++)
        _children.push_back(std::make_unique<Process>(comm::CommunicatorFactory::get_communicator(comm::EndpointFactory::get_endpoint(path)), _properties.plugins().at(i)));

    } else {
      LOG4CPLUS_ERROR(logger, "✖ - " << std::filesystem::path(__FILE__).filename() << ":" << __LINE__ << ":"
                                     << " json path error: no such file.");
      exit(EXIT_FAILURE);
    }
    LOG4CPLUS_DEBUG(logger, "✓ - [Process " << getpid() << "]: Finished.");
  }

  void
  Backend::Start() {
    // std::function<void(std::unique_ptr<gvirtus::Thread> & children)> task = [this](std::unique_ptr<gvirtus::Thread> &children) {
    //   LOG4CPLUS_DEBUG(logger, "✓ - [Thread " << std::this_thread::get_id() << "]: Started.");
    //   children->Start();
    //   LOG4CPLUS_DEBUG(logger, "✓ - [Thread " << std::this_thread::get_id() << "]: Finished.");
    // };

    int pid = 0;
    for (int i = 0; i < _children.size(); i++) {
      if ((pid = fork()) == 0) {
        _children[i]->Start();
        break;
      }
    }
    if (pid != 0) {
      int child_pid = 0;
      while ((child_pid = wait(nullptr)) > 0) {
        std::clog << __FILE__ << ":" << __LINE__ << ":"
                  << " Child process with this process id: " << child_pid << " has been terminated"
                  << ". ✓" << std::endl;
      }
    }

    if (pid == 0) {
      // CHILD
    }
  }

  void
  Backend::EventOccurred(std::string &event, void *object) {
    LOG4CPLUS_DEBUG(logger, "✓ - EventOccurred: " << event);
  }

} // namespace gvirtus