/* 
 * File:   Process.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.36
 */

#include <string>
#include "Process.h"

using namespace std;

Process::Process(const Communicator *communicator) 
    : mpInput(const_cast<Communicator *>(communicator)->GetInputStream()),
    mpOutput(const_cast<Communicator *>(communicator)->GetOutputStream()) {
    mpCommunicator = const_cast<Communicator *>(communicator);
}

Process::~Process() {
}

void Process::Setup() {
    
}

void Process::Execute(void * arg) {
    string routine;
    
    while(getline(mpInput, routine)) {
        if(routine.compare("ls") == 0)
            Ls();
        else 
            Default();
        mpOutput.flush();
    }
}

void Process::Default() {
    int result = -1;
    mpOutput.write((char *) &result, sizeof(int));
    result = 0;
    mpOutput.write((char *) &result, sizeof(int));
}

void Process::Ls() {
    int result = 0;
    mpOutput.write((char *) &result, sizeof(int));
    mpOutput.write((char *) &result, sizeof(int));
}
