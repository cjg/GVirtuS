#include "gvirtus/backend/Property.h"
#include <iostream>

using gvirtus::backend::Property;

Property &Property::endpoints(const int endpoints) {
  this->_endpoints = endpoints;
  return *this;
}

Property &Property::plugins(const std::vector<std::string> &plugins) {
  _plugins.emplace_back(plugins);
  return *this;
}

Property &Property::secure(bool secure) {
  _secure = secure;
  return *this;
}
