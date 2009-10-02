/* 
 * File:   Backend.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.29
 */

#include "Process.h"
#include "Backend.h"

Backend::Backend(const Communicator * communicator) {
    mpCommunicator = const_cast<Communicator *>(communicator);
}

Backend::~Backend() {
}

void Backend::Start() {
    mpCommunicator->Serve();
    while(true) {
        Communicator *client =
                const_cast<Communicator *>(mpCommunicator->Accept());
        Process *process = new Process(client);
        process->Start(NULL);
    }
}
