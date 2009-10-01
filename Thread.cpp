/* 
 * File:   Thread.cpp
 * Author: cjg
 * 
 * Created on October 1, 2009, 12:22 PM
 */

#include "Thread.h"

Thread::Thread() {
}

Thread::~Thread() {
}

int Thread::Start(void * arg) {
    mpArg = arg;
    return pthread_create(&mThreadId, NULL, Thread::EntryPoint, mpArg);
}

int Thread::Run(void * arg) {
    Setup();
    Execute(arg);
}

/*static */
void * Thread::EntryPoint(void * pthis) {
    Thread * pt = (Thread*) pthis;
    ((Thread *) pthis)->Run(((Thread *) pthis)->Arg());
}
