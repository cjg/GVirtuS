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
 * Written by: Roberto Di Lauro <roberto.dilauro@uniparthenope.it>,
 *             Department of Applied Science
 */


#ifndef _OCLHANDLER_H
#define	_OCLHANDLER_H

#include <iostream>
#include <map>
#include <string>
#include <cstdio>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "CudaUtil.h"
#include "CL/cl.h"
#include "Handler.h"

/**						
 * OclHandler is used by Backend's Process(es) for storing and retrieving
 * device related data and functions. 
 * OclHandler has also the method Execute() that is responsible to execute a
 * named OpenCL routine unmarshalling the input parameters from the
 * provided Buffer.
 */
class OclHandler : public Handler {
public:
    OclHandler();
    virtual ~OclHandler();
    Result * Execute(std::string routine, Buffer * input_buffer);

    void* RegisterPointer(void *,size_t);

    void RegisterMapObject(char *,char *);
    char * GetMapObject(char *);
    

private:
    void Initialize();
    typedef Result * (*OclRoutineHandler)(OclHandler *, Buffer *);
    static std::map<std::string, OclRoutineHandler> * oclHandlers;
    void **pointers;
    int nPointers;
    std::map<std::string, std::string> * mpMapObject;
    
    void *mpShm;
    int mShmFd;
};

#define OCL_ROUTINE_HANDLER(name) Result * handle##name(OclHandler * pThis, Buffer * input_buffer)
#define OCL_ROUTINE_HANDLER_PAIR(name) make_pair("cl" #name, handle##name)

/* OclHandler Query Platform Info */
OCL_ROUTINE_HANDLER(GetPlatformIDs);
OCL_ROUTINE_HANDLER(GetPlatformInfo);

/* OclHandler Query Devices  */
OCL_ROUTINE_HANDLER(GetDeviceIDs);
OCL_ROUTINE_HANDLER(GetDeviceInfo);


/* OclHandler Contexts  */
OCL_ROUTINE_HANDLER(CreateContext);
OCL_ROUTINE_HANDLER(GetContextInfo);

/* OclHandler Memory Objects */
OCL_ROUTINE_HANDLER(GetSupportedImageFormats);
OCL_ROUTINE_HANDLER(CreateBuffer);


OCL_ROUTINE_HANDLER(CreateCommandQueue);

OCL_ROUTINE_HANDLER(CreateProgramWithSource);
OCL_ROUTINE_HANDLER(BuildProgram);
OCL_ROUTINE_HANDLER(CreateKernel);
OCL_ROUTINE_HANDLER(EnqueueCopyBuffer);
OCL_ROUTINE_HANDLER(SetKernelArg);

OCL_ROUTINE_HANDLER(EnqueueNDRangeKernel);
OCL_ROUTINE_HANDLER(Flush);
OCL_ROUTINE_HANDLER(Finish);
OCL_ROUTINE_HANDLER(EnqueueReadBuffer);
OCL_ROUTINE_HANDLER(EnqueueWriteBuffer);
OCL_ROUTINE_HANDLER(WaitForEvents);
OCL_ROUTINE_HANDLER(ReleaseMemObject);
OCL_ROUTINE_HANDLER(ReleaseEvent);
OCL_ROUTINE_HANDLER(ReleaseKernel);
OCL_ROUTINE_HANDLER(ReleaseCommandQueue);
OCL_ROUTINE_HANDLER(ReleaseProgram);
OCL_ROUTINE_HANDLER(ReleaseContext);
OCL_ROUTINE_HANDLER(GetProgramInfo);
OCL_ROUTINE_HANDLER(EnqueueMapBuffer);
OCL_ROUTINE_HANDLER(EnqueueUnmapMemObject);
OCL_ROUTINE_HANDLER(GetKernelWorkGroupInfo);
OCL_ROUTINE_HANDLER(GetCommandQueueInfo);
OCL_ROUTINE_HANDLER(GetEventProfilingInfo);




#endif	/* _OCLHANDLER_H */

