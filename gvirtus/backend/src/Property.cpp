#include "Property.h"
#include <filesystem>
#include <iostream>

namespace gvirtus {

  Property &
  Property::endpoints(const int endpoints) {
    this->_endpoints = endpoints;
    return *this;
  }

  Property &
  Property::plugins(const std::vector<std::string> &plugins) {
    _plugins.emplace_back(plugins);
    return *this;
  }
} // namespace gvirtus