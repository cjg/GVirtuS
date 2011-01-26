/* 
 * File:   OclProcess.h
 * Author: roberto
 *
 * Created on January 25, 2011, 6:59 PM
 */

#include "Process.h"
#include <iostream>
#include <cstdio>
#include <string>


#ifndef OCLPROCESS_H
#define	OCLPROCESS_H

using namespace std;


class OclProcess : public Process {
public:
    OclProcess(const Communicator *communicator, Handler *handler);
    ~OclProcess();
    void Execute(void * arg);

private:

};

#endif	/* OCLPROCESS_H */

