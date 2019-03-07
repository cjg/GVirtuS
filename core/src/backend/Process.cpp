#include <backend/Process.h>

//Fork library
#include <sys/types.h>
#include <unistd.h>
#include <iostream>

namespace gvirtus {
    Process::Process(std::unique_ptr<Communicator> comm) {
        _communicator = std::move(comm);
    }


    void Process::start() {
        std::clog << __FILE__ << ":" << __LINE__ << ":" << " Process with this process id: " << getpid() << " started. ✓" << std::endl;
        _communicator->run();
    }
}
