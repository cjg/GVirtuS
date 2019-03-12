#include <backend/Backend.h>
#include <communicator/Communicator.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <util/JSON.h>

//for getpid()
#include <unistd.h>

namespace gvirtus {
    Backend::Backend(const std::filesystem::path &path) {
        if (std::filesystem::exists(path) && std::filesystem::is_regular_file(path) && path.extension() == ".json") {
            _json_path = path;
        } else {
            std::cerr << __FILE__ << ":" << __LINE__ << ":" << " Path error: no such file. ✗" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::clog << __FILE__ << ":" << __LINE__ << ":" << " Json file has been loaded. ✓" << std::endl;

        _property = gvirtus::util::JSON<Property>(_json_path).parser();
        for (int i = 0; i < _property.endpoints(); i++)
            _children.emplace_back(Process(std::make_unique<Communicator>(gvirtus::Endpoint(_json_path))));

        std::clog << __FILE__ << ":" << __LINE__ << ":" << " Processes and endpoints have been associated. ✓"
                  << std::endl;

        init();
    }

    void Backend::init() {
        // FIXME: dove devono essere caricati i plugin?
        std::clog << __FILE__ << ":" << __LINE__ << ":" << " Plugins loaded. ✓" << std::endl;
    }

    void Backend::start() {
        int pid = 0;
        for (int i = 0; i < _children.size(); i++) {
            if ((pid = fork()) == 0) {
                _children[i].start();
                break;
            }
            std::clog << __FILE__ << ":" << __LINE__ << ":" << " Child process created with this process id: "
                      << pid << ". ✓" << std::endl;
        }

        if (pid != 0) {
            int child_pid = 0;
            while ((child_pid = wait(nullptr)) > 0) {
                std::clog << __FILE__ << ":" << __LINE__ << ":" << " Child process with this process id: " << child_pid
                          << " has been terminated" << ". ✓" << std::endl;
            }

//            std::clog << __FILE__ << ":" << __LINE__ << ":" << " Father process with this process id: " << getpid()
//                      << " has been terminated. ✓" << std::endl;
        }

        if (pid == 0) {
            //CHILD
        }
    }
}