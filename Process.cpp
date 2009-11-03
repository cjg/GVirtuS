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
    while(getline(mpInput, routine)) {
        Buffer * input_buffer = new Buffer(mpInput);
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
        delete input_buffer;
        result->Dump(mpOutput);
        mpOutput.flush();
        cout << "[Process " << GetThreadId() << "]: Exit Code '"
            << cudaGetErrorString(result->GetExitCode()) << "'." << endl;
        delete result;
    }
    Notify("process-ended");
    delete this;
}

