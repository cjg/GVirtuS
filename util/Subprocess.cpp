/* 
 * File:   Subprocess.cpp
 * Author: cjg
 * 
 * Created on December 23, 2009, 8:56 PM
 */

#include <sys/wait.h>
#include <stdlib.h>
#include <csignal>
#include "Subprocess.h"

Subprocess::Subprocess() {
    signal(SIGCHLD, SIG_IGN);
}

Subprocess::~Subprocess() {
}

int Subprocess::Start(void * arg) {
    mpArg = arg;
    if((mPid = fork()) < 0)
        throw "Can't instantiate Subprocess.";
    if(mPid == 0) {
        this->EntryPoint(this);
        exit(0);
    }
    return mPid;
}

void Subprocess::Wait() {
    int status;
    waitpid(mPid, &status, 0);
}

int Subprocess::Run(void * arg) {
    Setup();
    Execute(arg);
    return 0;
}

/*static */
void * Subprocess::EntryPoint(void * pthis) {
    ((Subprocess *) pthis)->Run(((Subprocess *) pthis)->mpArg);
    return NULL;
}

pid_t Subprocess::GetPid() {
    if(mPid == 0)
        mPid = getpid();
    return mPid;
}

