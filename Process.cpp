/* 
 * File:   Process.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.36
 */

#include <iostream>
#include <cstdio>
#include <string>
#include "cuda_runtime_api.h"
#include "Process.h"

using namespace std;

Process::Process(const Communicator *communicator) 
    : Thread(), Observable(), 
        mpInput(const_cast<Communicator *>(communicator)->GetInputStream()),
        mpOutput(const_cast<Communicator *>(communicator)->GetOutputStream()) {
    mpCommunicator = const_cast<Communicator *>(communicator);
    mpHandler = new CudaRtHandler();
    cout << "[Process ]: Created." << endl;
}

Process::~Process() {
    cout << "[Process " << GetThreadId() <<  "]: Destroyed." << endl;
}

void Process::Setup() {
    
}

void Process::Execute(void * arg) {
    cout << "[Process " << GetThreadId() <<  "]: Started." << endl;

    string routine;
    Buffer * input_buffer = new Buffer();
    while(getline(mpInput, routine)) {
        //Buffer * input_buffer = new Buffer(mpInput);
        input_buffer->Reset(mpInput);
        Result * result;
        cout << "[Process " << GetThreadId() <<  "]: Requested '" << routine
            << "' routine." << endl;
        try {
             result = mpHandler->Execute(routine, input_buffer);
        } catch(string e) {
            cout << "[Process " << GetThreadId() <<  "]: Exception " << e
                << "." << endl;
            result = new Result(cudaErrorUnknown, new Buffer());
        }
        result->Dump(mpOutput);
        cout << "[Process " << GetThreadId() << "]: Exit Code '"
            << cudaGetErrorString(result->GetExitCode()) << "'." << endl;
        delete result;
    }
    delete input_buffer;
    Notify("process-ended");
    delete this;
}

