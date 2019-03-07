#ifndef GVIRTUSPP_PROCESS_H
#define GVIRTUSPP_PROCESS_H

// Temporarily
#include <string>
#include <communicator/Communicator.h>

namespace gvirtus {
    /**
     * Process class.
     */
    class Process {
    public:
        Process() = default;

        explicit Process(std::unique_ptr<Communicator>);

        void start();

    private:
        std::unique_ptr<Communicator> _communicator;
    };
}

#endif //GVIRTUSPP_PROCESS_H
