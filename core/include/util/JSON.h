#ifndef GVIRTUSPP_JSON_H
#define GVIRTUSPP_JSON_H

#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace gvirtus {
    /**
     * JSON class.
     * This class is a wrapper of nlohmann::json class.
     * When you are starting this class you must indicate a type of class, here called T.
     * This serves to parser method to determine how to construct an object from a json file or as
     * a json file to build from a given class.
     * @tparam T: T must be a class. This class must have the functions to_json and from_json defined. See Property.h for more info
     */
    template<typename T>
    class JSON {
    public:
        /**
         * Default constructor
         */
        JSON() = default;

        /**
         * Parameterized constructor
         * @param file_path: the path where json file is located
         */
        //TOdO: default path?
        JSON(std::filesystem::path file_path) {
            path(file_path);
        }

        /**
         * This method loads the contents of the json file into the json object using a std::fstream file handler
         * @param file_path: the path where json file is located
         * @return reference to itself (Fluent Interface API)
         */
        //TOdO: default path?
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

        /**
         * @return the path where json file is located
         */
        inline const std::filesystem::path path() const {
            return _file_path;
        }

        /**
         * @return copy of the object built from json file
         */
        T parser() {
            if (!_json.is_null()) {
                auto object = _json.get<T>();
                return object;
            }

            return T();
        }

        /**
         *  Class destroyer
         */
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
