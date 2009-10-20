/* 
 * File:   Frontend.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.57
 */

#ifndef _FRONTEND_H
#define	_FRONTEND_H

#include <builtin_types.h>
#include "Communicator.h"
#include "Result.h"

class Frontend {
public:
    virtual ~Frontend();
    static Frontend & GetFrontend();
    Result * Execute(const char *routine, const Buffer *input_buffer);
private:
    Frontend();
    Communicator *mpCommunicator;
    static Frontend *mspFrontend;
};

#endif	/* _FRONTEND_H */

