#ifndef GVIRTUSPP_ENDPOINTEXCEPTION_H
#define GVIRTUSPP_ENDPOINTEXCEPTION_H

#include <exception/Exception.h>

namespace gvirtus {
    class EndpointException : public gvirtus::Exception {
    public:
        EndpointException(const std::string file, int line, const std::string func, const std::string info = "")
                : gvirtus::Exception{file, line, func, info} {
        }
    };
}

#endif //GVIRTUSPP_ENDPOINTEXCEPTION_H
