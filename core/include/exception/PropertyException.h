#ifndef GVIRTUSPP_PROPERTYEXCEPTION_H
#define GVIRTUSPP_PROPERTYEXCEPTION_H

#include <exception/Exception.h>

namespace gvirtus {
    class PropertyException : public gvirtus::Exception {
    public:
        PropertyException(const std::string file, int line, const std::string func, const std::string info = "")
                : gvirtus::Exception(_file, _line, _func, _info) {
        }

    };
}
#endif //GVIRTUSPP_PROPERTYEXCEPTION_H
