#ifndef GVIRTUS_PROPERTY_H
#define GVIRTUS_PROPERTY_H

#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace gvirtus {
/**
 * Property class.
 * This class is a model to represent the properties of the backend.
 */
class Property {
 public:
  /**
   * Default constructor
   */
  Property() = default;

  /**
   * This method is a setter for the class member _endpoints
   * @param endpoints: pointer to a vector of Endpoint
   * @return reference to itself (Fluent Interface API)
   */
  Property &endpoints(const int endpoints);

  /**
   * This method is a getter for the class member _endpoints
   * @return the reference to vector where Endpoint are saved
   */
  inline const int &
  endpoints() const {
    return _endpoints;
  }

  /**
   * This method is a setter for the class member _plugins
   * @param plugins: pointer to a vector of std::string
   * @return reference to itself (Fluent Interface API)
   */
  Property &plugins(const std::vector<std::string> &plugins);

  /**
   * This method is a getter for the class member _plugins
   * @return the reference to vector where plugins string are saved
   */
  inline std::vector<std::vector<std::string>> &
  plugins() {
    return _plugins;
  }

  Property &secure(bool secure);

  inline bool &
  secure() {
    return _secure;
  }

 private:
  std::vector<std::vector<std::string>> _plugins;
  int _endpoints;
  bool _secure;
};

/**
 * This function will be used by nlohmann::json object when we call
 * j.get&lt;Property&lt;(). Without this function the program doesn't know how
 * to build a property object from json object.
 * @param j: reference to json object which contains the data
 * @param p: reference to property object to be created
 */
inline void
from_json(const nlohmann::json &j, Property &p) {
  int ends = 0;
  for (auto &el : j["communicator"]) {
    ends++;
    p.plugins(el["plugins"].get<std::vector<std::string>>());
  }

  p.endpoints(ends);
  p.secure(j["secure_application"].get<bool>());
}
} // namespace gvirtus
#endif // GVIRTUS_PROPERTY_H
