/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>,
 *             Department of Science and Technologies
 */

#ifndef _CURANDHANDLER_H
#define _CURANDHANDLER_H

#include "Handler.h"
#include "Result.h"
//#include "CudaUtil.h"
#include <curand.h>

#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
#include "log4cplus/configurator.h"

class CurandHandler : public Handler {
public:
    CurandHandler();
    virtual ~CurandHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);

    /*void * RegisterPointer(void *,size_t);

    void RegisterMapObject(char *,char *);
    char * GetMapObject(char *);
    */
private:
    log4cplus::Logger logger;
    void Initialize();
    typedef Result * (*CurandRoutineHandler)(CurandHandler *, Buffer *);
    static std::map<std::string, CurandRoutineHandler> * mspHandlers;
    //void **pointers;
    //int nPointers;
    
    //std::map<std::string, std::string> * mpMapObject;

    //void *mpShm;
    //int mShmFd;
};

#define CURAND_ROUTINE_HANDLER(name) Result * handle##name(CurandHandler * pThis, Buffer * in)
#define CURAND_ROUTINE_HANDLER_PAIR(name) make_pair("curand" #name, handle##name)

/* CudnnHandler_Platform */
//CUDNN_ROUTINE_HANDLER(GetVersion);
#endif //_CURANDHANDLER_H