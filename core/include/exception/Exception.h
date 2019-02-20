//
// Created by antonio on 20.02.19.
//

#ifndef GVIRTUSPP_EXCEPTION_H
#define GVIRTUSPP_EXCEPTION_H

#include <stdexcept>

namespace gvirtus {
    class Exception : public std::exception {
    public:
        Exception(const std::string file, int line, const std::string func, const std::string info = "")
                : std::exception(),
                  _file(file),
                  _line(line),
                  _func(func),
                  _info(info) {
        }

        const std::string get_file() const { return _file; }

        int get_line() const { return _line; }

        const std::string get_func() const { return _func; }

        const std::string get_info() const { return _info; }

    protected:
        const std::string _file;
        int _line;
        const std::string _func;
        const std::string _info;
    };
}
#endif //GVIRTUSPP_EXCEPTION_H
