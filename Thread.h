/* 
 * File:   Thread.h
 * Author: cjg
 *
 * Created on October 1, 2009, 12:22 PM
 */

#ifndef _THREAD_H
#define	_THREAD_H

#include <pthread.h>

class Thread {
public:
    Thread();
    virtual ~Thread();
    int Start(void * arg);
protected:
    int Run(void * arg);
    static void * EntryPoint(void*);
    virtual void Setup();
    virtual void Execute(void*);
    pthread_t GetThreadId();
    void * Arg() const {return mpArg;}
    void Arg(void* a){mpArg = a;}
private:
    pthread_t mThreadId;
    void * mpArg;
};

#endif	/* _THREAD_H */

