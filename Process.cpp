/* 
 * File:   Process.cpp
 * Author: cjg
 * 
 * Created on 30 settembre 2009, 12.36
 */

#include <string>
#include "Process.h"

using namespace std;

Process::Process(const Communicator *communicator) {
    mpCommunicator = const_cast<Communicator *>(communicator);
}

Process::~Process() {
}

void Process::Start() {
    string routine;
    
    while(getline(mpCommunicator->GetInputStream(), routine)) {
        cout << routine << endl;
    }
}
