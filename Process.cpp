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
: Subprocess(), Observable(),
mpInput(const_cast<Communicator *> (communicator)->GetInputStream()),
mpOutput(const_cast<Communicator *> (communicator)->GetOutputStream()) {
    mpCommunicator = const_cast<Communicator *> (communicator);
    mpHandler = new CudaRtHandler();
}

Process::~Process() {
    cout << "[Process " << GetPid() << "]: Destroyed." << endl;
}

void Process::Setup() {

}

void Process::Execute(void * arg) {
    cout << "[Process " << GetPid() << "]: Started." << endl;

    string routine;
    Buffer * input_buffer = new Buffer();
    while (getline(mpInput, routine)) {
        input_buffer->Reset(mpInput);
        Result * result;
        try {
            result = mpHandler->Execute(routine, input_buffer);
        } catch (string e) {
            cout << "[Process " << GetPid() << "]: Exception " << e
                    << "." << endl;
            result = new Result(cudaErrorUnknown, new Buffer());
        }
        result->Dump(mpOutput);
        if (result->GetExitCode() != cudaSuccess) {
            cout << "[Process " << GetPid() << "]: Requested '" << routine
                    << "' routine." << endl;
            cout << "[Process " << GetPid() << "]: Exit Code '"
                    << cudaGetErrorString(result->GetExitCode()) << "'." << endl;
        }
        delete result;
    }
    delete input_buffer;
    Notify("process-ended");
    delete this;
}

