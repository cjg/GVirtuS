/* 
 * File:   Backend.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.29
 */

#include <iostream>
#include "Process.h"
#include "Backend.h"

Backend::Backend(const Communicator * communicator) : Observer() {
    mpCommunicator = const_cast<Communicator *>(communicator);
}

void Backend::Start() {
    mpCommunicator->Serve();
    while(true) {
        Communicator *client =
                const_cast<Communicator *>(mpCommunicator->Accept());
        Process *process = new Process(client);
        process->AddObserver("process-ended", this);
        process->Start(NULL);
    }
}

void Backend::EventOccurred(std::string& event, void* object) {
    std::cout << "EventOccurred: " << event << std::endl;
    if(event.compare("process-ended") == 0) {

    }
}
