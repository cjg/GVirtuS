#ifndef GVIRTUS_LD_LIB_H
#define GVIRTUS_LD_LIB_H

#include "log4cplus/configurator.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include <dlfcn.h>
#include <memory>
#include <string>
#include <filesystem>

template <class T, typename... Args> class LD_Lib {
private:
  using create_t = std::shared_ptr<T>(Args ...);

  void
  dlopen(const std::string path, int flag = RTLD_LAZY) {
    m_module = ::dlopen(path.c_str(), flag);

    if (m_module == nullptr)
      throw std::string("Error loading: " + std::string(dlerror()));
  }

  void
  set_create_symbol(const std::string &function) {
    sym = (create_t *)dlsym(m_module, function.c_str());

    if (sym == nullptr) {
      ::dlclose(m_module);
      throw std::string("Cannot load symbol create: " + std::string(dlerror()) + "\n");
    }

  }

public:
  LD_Lib(const std::string path, std::string fcreate_name = "create_t") {
    logger = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("LD_Lib"));
    _obj_ptr = nullptr;
    dlopen(path);
    set_create_symbol(fcreate_name);
    LOG4CPLUS_DEBUG(logger, "✓ - Loaded module " << std::filesystem::path(path).filename() << ".");
  }

  ~LD_Lib() {
    if(m_module != nullptr) {
      sym = nullptr;
      _obj_ptr.reset();
      ::dlclose(m_module);
    }
  }

  void build_obj(Args ... args) {
    _obj_ptr = this->sym(args...);
  }

  std::shared_ptr<T> obj_ptr(){
    return _obj_ptr;
  }

protected:
  create_t *sym;
  void *m_module;
  std::shared_ptr<T> _obj_ptr;
  log4cplus::Logger logger;
};

#endif // GVIRTUS_LD_LIB_H