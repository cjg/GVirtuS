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
        _communicator->run();
    }
}
