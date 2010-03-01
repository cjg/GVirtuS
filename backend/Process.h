/* 
 * File:   Process.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.36
 */

#ifndef _PROCESS_H
#define	_PROCESS_H

#include "Subprocess.h"
#include "Observable.h"
#include "Communicator.h"
#include "CudaRtHandler.h"

class Process : public Subprocess, public Observable {
public:
    Process(const Communicator *communicator);
    virtual ~Process();
    void Setup();
    void Execute(void * arg);
private:
    Communicator * mpCommunicator;
    std::istream & mpInput;
    std::ostream & mpOutput;
    CudaRtHandler * mpHandler;
};

#endif	/* _PROCESS_H */

