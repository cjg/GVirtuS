/* 
 * File:   Backend.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.29
 */

#ifndef _BACKEND_H
#define	_BACKEND_H

#include "Communicator.h"

class Backend {
public:
    Backend();
    virtual ~Backend();
    void Start();
private:
    Communicator *mpCommunicator;
};

#endif	/* _BACKEND_H */

