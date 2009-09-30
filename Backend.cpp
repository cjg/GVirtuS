/* 
 * File:   Backend.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.29
 */

#include "AfUnixCommunicator.h"
#include "Backend.h"

Backend::Backend() {
    // TODO: Communicator to use should be obtained from config file
    mpCommunicator = new AfUnixCommunicator("/tmp/cudactl");
}

Backend::~Backend() {
}

void Backend::Start() {
    mpCommunicator->Serve();
    while(true) {
        Communicator *client = mpCommunicator->Accept();
        // Process *process = new Process(client);
        // process->Start();
    }
}
