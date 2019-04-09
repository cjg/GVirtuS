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
 * Written by: Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>,
 *             Department of Applied Science
 */

/**
 * @file   CudaRtHandler.h
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 * 
 * @brief  
 * 
 * 
 */



#include <iostream>
#include <map>
#include <string>
#include <cstdio>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <CL/cl.h>

#include "Handler.h"
#include "Result.h"

#include "OpenclUtil.h"

class OpenclHandler : public Handler {
public:
    OpenclHandler();
    virtual ~OpenclHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);


    void* RegisterPointer(void *,size_t);

    void RegisterMapObject(char *,char *);
    char * GetMapObject(char *);


private:
    void Initialize();
    typedef Result * (*OpenclRoutineHandler)(OpenclHandler *, Buffer *);
    static std::map<std::string, OpenclRoutineHandler> * mspHandlers;
    void **pointers;
    int nPointers;
    std::map<std::string, std::string> * mpMapObject;

    void *mpShm;
    int mShmFd;
};

#define OPENCL_ROUTINE_HANDLER(name) Result * handle##name(OpenclHandler * pThis, Buffer * input_buffer)
#define OPENCL_ROUTINE_HANDLER_PAIR(name) make_pair("cl" #name, handle##name)

/* OpenclHandler_Platform */
OPENCL_ROUTINE_HANDLER(GetPlatformIDs);
OPENCL_ROUTINE_HANDLER(GetDeviceIDs);
OPENCL_ROUTINE_HANDLER(CreateContext);
OPENCL_ROUTINE_HANDLER(CreateCommandQueue);
OPENCL_ROUTINE_HANDLER(CreateProgramWithSource);
OPENCL_ROUTINE_HANDLER(BuildProgram);
OPENCL_ROUTINE_HANDLER(CreateBuffer);
OPENCL_ROUTINE_HANDLER(CreateKernel);
OPENCL_ROUTINE_HANDLER(SetKernelArg);
OPENCL_ROUTINE_HANDLER(EnqueueWriteBuffer);
OPENCL_ROUTINE_HANDLER(EnqueueNDRangeKernel);
OPENCL_ROUTINE_HANDLER(EnqueueReadBuffer);
OPENCL_ROUTINE_HANDLER(Finish);
OPENCL_ROUTINE_HANDLER(GetPlatformInfo);
OPENCL_ROUTINE_HANDLER(GetContextInfo);
OPENCL_ROUTINE_HANDLER(Flush);
OPENCL_ROUTINE_HANDLER(EnqueueCopyBuffer);
OPENCL_ROUTINE_HANDLER(ReleaseMemObject);
OPENCL_ROUTINE_HANDLER(ReleaseKernel);
OPENCL_ROUTINE_HANDLER(ReleaseContext);
OPENCL_ROUTINE_HANDLER(ReleaseCommandQueue);
OPENCL_ROUTINE_HANDLER(ReleaseProgram);
//OPENCL_ROUTINE_HANDLER(ReleaseEvent);
//OPENCL_ROUTINE_HANDLER(WaitForEvents);

