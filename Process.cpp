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
        if(routine.compare("ls") == 0)
            Ls();
        else 
            Default();
        mpOutput.flush();
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

void Process::Ls() {
    cout << "[Process " << GetThreadId() <<  "]: Executing Ls()." << endl;
    int result = 0;
    mpOutput.write((char *) &result, sizeof(int));
    mpOutput.write((char *) &result, sizeof(int));
}
