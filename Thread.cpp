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
    return pthread_create(&mThreadId, NULL, Thread::EntryPoint, this);
}

void Thread::Join() {
    pthread_join(mThreadId, NULL);
}

int Thread::Run(void * arg) {
    Setup();
    Execute(arg);
    return 0;
}

/*static */
void * Thread::EntryPoint(void * pthis) {
    Thread * pt = (Thread*) pthis;
    ((Thread *) pthis)->Run(((Thread *) pthis)->mpArg);
}

pthread_t Thread::GetThreadId() {
    return mThreadId;
}
