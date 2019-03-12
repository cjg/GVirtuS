#ifndef GVIRTUSPP_BACKEND_H
#define GVIRTUSPP_BACKEND_H

#include <backend/Property.h>
#include <backend/Process.h>
#include <vector>

namespace gvirtus {
    /**
     * Backend class.
     */
    class Backend {
    public:
        /**
         * Parameterized constructor
         * @param p:
         */
        explicit Backend(const std::filesystem::path &);

        void start();

    private:
        std::filesystem::path _json_path;
        std::vector<Process> _children;
        Property _property;

        void init();
    };
}

#endif //GVIRTUSPP_BACKEND_H
