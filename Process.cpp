/* 
 * File:   Process.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.36
 */

#include <iostream>
#include <string>
#include "Process.h"

using namespace std;

Process::Process(const Communicator *communicator) 
    : Thread(), Observable(), 
        mpInput(const_cast<Communicator *>(communicator)->GetInputStream()),
        mpOutput(const_cast<Communicator *>(communicator)->GetOutputStream()) {
    mpCommunicator = const_cast<Communicator *>(communicator);
    mpHandler = new CudaRtHandler();
    cout << "[Process " << GetThreadId() <<  "]: Created." << endl;
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
        cout << "[Process " << GetThreadId() <<  "]: Requested '" << routine
            << "' routine." << endl;

        char *in_buffer, *out_buffer;
        size_t in_buffer_size, out_buffer_size;
        mpInput.read((char *) &in_buffer_size, sizeof(size_t));
        in_buffer = new char[in_buffer_size];
        mpInput.read(in_buffer, in_buffer_size);

        cudaError_t result;
        try {
             result = mpHandler->Execute(routine, in_buffer,
                    in_buffer_size, &out_buffer, &out_buffer_size);
        } catch(string e) {
            cout << "[Process " << GetThreadId() <<  "]: Exception " << e
                << "." << endl;
            delete[] in_buffer;
            result = cudaErrorUnknown;
            mpOutput.write((char *) &result, sizeof(cudaError_t));
            mpOutput.flush();
            continue;
        }

        delete[] in_buffer;

        mpOutput.write((char *) &result, sizeof(cudaError_t));
        if(result == cudaSuccess) {
            mpOutput.write((char *) &out_buffer_size, sizeof(size_t));
            mpOutput.write(out_buffer, out_buffer_size);
            delete[] out_buffer;
        }

        mpOutput.flush();
        cout << "[Process " << GetThreadId() << "]: Result " << result << "."
            << endl;
    }
    Notify("process-ended");
    delete this;
}

void Process::Default() {
    cout << "[Process " << GetThreadId() <<  "]: Executing Default()." << endl;
    int result = -1;
    mpOutput.write((char *) &result, sizeof(int));
    result = 0;
    mpOutput.write((char *) &result, sizeof(int));
}

