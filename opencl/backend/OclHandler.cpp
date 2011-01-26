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

/**
 * @file   Backend.cpp
 * @author Roberto Di Lauro <roberto.dilauro@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 *
 * @brief
 *
 *
 */

#include "OclHandler.h"

#include <cstring>

#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string.h>

#include "CudaUtil.h"

using namespace std;

map<string, OclHandler::OclRoutineHandler> *OclHandler::oclHandlers = NULL;


OclHandler::OclHandler() {
    mpMapObject = new map<string, string> ();
    pointers = (void **)malloc(sizeof(void*));
    mpMapObject = new map<string, string>();
    nPointers = 1;
    
    Initialize();
}

OclHandler::~OclHandler() {
    
}

Result * OclHandler::Execute(std::string routine, Buffer * input_buffer) {
    map<string, OclHandler::OclRoutineHandler>::iterator it;
    it = oclHandlers->find(routine);
    if (it == oclHandlers->end())
        throw "No handler for '" + routine + "' found!";
    return it->second(this, input_buffer);
}

void* OclHandler::RegisterPointer(void* pointer,size_t bytes){
    if (nPointers==1){
        pointers[0] = (char *)malloc(bytes);
        memcpy(pointers[0],(char *)pointer,bytes);
        nPointers = nPointers + 1;
        return pointers[0];
    }else{
        pointers = (void**)realloc(pointers,nPointers * sizeof(void*));
        pointers[nPointers-1] = (char *)malloc(bytes);
        memcpy(pointers[nPointers-1],pointer,bytes);
        nPointers = nPointers + 1;
        return pointers[nPointers-2];
    }
}

    void OclHandler::RegisterMapObject(char * key,char * value){
        
        mpMapObject->insert(make_pair(key, value));

    }
    char * OclHandler::GetMapObject(char * key){
        for (map<string, string>::iterator it = mpMapObject->begin();
            it != mpMapObject->end(); it++)
        if (it->first == key)
            return (char *)(it->second.c_str());
    return NULL;
    }

void OclHandler::Initialize() {
    if (oclHandlers != NULL)
        return;
    pointers = (void **)malloc(sizeof(void*));
    mpMapObject = new map<string, string>();
    nPointers = 1;
    oclHandlers = new map<string, OclHandler::OclRoutineHandler> ();

    /* OclHandler Query Platform Info */
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetPlatformIDs));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetPlatformInfo));

    /* OclHandler Query Devices  */
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetDeviceIDs));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetDeviceInfo));

    /* OclHandler Context*/
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(CreateContext));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetContextInfo));

    /* OclHandler Memory Objects*/
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetSupportedImageFormats));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(CreateBuffer));


    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(CreateCommandQueue));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(CreateProgramWithSource));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(BuildProgram));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(CreateKernel));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(EnqueueCopyBuffer));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(SetKernelArg));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(EnqueueNDRangeKernel));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(Flush));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(Finish));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(EnqueueReadBuffer));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(EnqueueWriteBuffer));


    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(WaitForEvents));

    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(ReleaseMemObject));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(ReleaseEvent));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(ReleaseKernel));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(ReleaseCommandQueue));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(ReleaseProgram));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(ReleaseContext));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetProgramInfo));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(EnqueueMapBuffer));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(EnqueueUnmapMemObject));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetKernelWorkGroupInfo));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetCommandQueueInfo));
    oclHandlers->insert(OCL_ROUTINE_HANDLER_PAIR(GetEventProfilingInfo));




}
