#include <backend/Property.h>
#include <iostream>
#include <filesystem>

namespace gvirtus {
    Property::Property(const int *endpoints_required,
                       const std::vector<std::string> *plugins_required) {
        endpoints(endpoints_required);
        plugins(plugins_required);
    }

    Property &Property::endpoints(const int *endpoints) {
        if (endpoints == nullptr || *endpoints == 0) {
//            _endpoints.emplace_back(gvirtus::Endpoint("tcp", "127.0.0.1", "9999"));
        } else
            _endpoints = *endpoints;

        return *this;
    }

    /**
     * This function must be used server side.
     * @param plugins
     * @return
     */
    Property &Property::plugins(const std::vector<std::string> *plugins) {
        namespace fs = std::filesystem;
        fs::path p(fs::current_path().parent_path());
        p += "/core/plugins";

        if (fs::exists(p) && fs::is_directory(p) && !fs::is_empty(p)) {
            for (const auto &entry : fs::directory_iterator(p)) {
                if (fs::is_directory(entry.status())) {
                    auto filename = entry.path().filename().string();
                    for (const auto &plugin : *plugins) {
                        if (filename == plugin)
                            _plugins.emplace_back(filename);
                    }
                }
            }

            //TODO: exception or optional?
            if (_plugins.empty())
                std::cerr << "Plugin not found" << std::endl;
        } else
            //TODO: exception or optional?
            std::cerr << "Plugins: no such directory, or is empty." << std::endl;

        return *this;
    }
}