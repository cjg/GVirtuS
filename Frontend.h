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

class Frontend {
public:
    virtual ~Frontend();
    static Frontend & GetFrontend();
    cudaError_t Execute(const char *routine, const char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size);
private:
    Frontend();
    Communicator *mpCommunicator;
    static Frontend *mspFrontend;
};

#endif	/* _FRONTEND_H */

