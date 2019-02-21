#ifndef GVIRTUSPP_JSON_H
#define GVIRTUSPP_JSON_H

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace gvirtus {
    template<typename T>
    class JSON {
    public:
        JSON() = default;

        JSON(std::filesystem::path file_path) {
            path(file_path);
        }

        JSON &path(std::filesystem::path &file_path) {
            _handler.exceptions(std::ios::failbit | std::ios::badbit);
            namespace fs = std::filesystem;

            if (fs::exists(file_path) && fs::is_regular_file(file_path) &&
                file_path.filename().extension() == ".json") {

                _handler.open(file_path.string());

                if (_handler.is_open()) {
                    _handler >> _json;
                } else
                    throw std::ifstream::failure("Can't open file");
            } else
                throw std::ifstream::failure("No such file, or file is irregular");

            return *this;
        }

        inline const std::filesystem::path path() const {
            return _file_path;
        }

        T parser() {
            if (!_json.is_null()) {
                auto object = _json.get<T>();
                return object;
            }

            return T();
        }

        ~JSON() {
            _json.clear();
            _handler.close();
            _file_path.clear();
        }
    private:
        nlohmann::json _json;
        std::ifstream _handler;
        std::filesystem::path _file_path;
    };
}

#endif //GVIRTUSPP_JSON_H
