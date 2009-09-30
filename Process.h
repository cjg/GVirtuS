/* 
 * File:   Process.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.36
 */

#ifndef _PROCESS_H
#define	_PROCESS_H

#include "Communicator.h"

class Process {
public:
    Process(const Communicator *communicator);
    virtual ~Process();
    void Start();
private:
    void Ls();
    Communicator *mpCommunicator;
    std::istream *mpInput;
    std::ostream *mpOutput;
};

#endif	/* _PROCESS_H */

