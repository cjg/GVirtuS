/* 
 * File:   Frontend.h
 * Author: cjg
 *
 * Created on 30 settembre 2009, 12.57
 */

#ifndef _FRONTEND_H
#define	_FRONTEND_H

#include "Communicator.h"

class Frontend {
public:
    virtual ~Frontend();
    static Frontend & GetFrontend();
    int Execute(const char *routine, const char *in_buffer, 
        int in_buffer_size, char **out_buffer, int *out_buffer_size);
private:
    Frontend();
    Communicator *mpCommunicator;
    static Frontend *mspFrontend;

};

Frontend *Frontend::mspFrontend = NULL;

#endif	/* _FRONTEND_H */

