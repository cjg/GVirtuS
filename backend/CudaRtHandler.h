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

#ifndef _CUDARTHANDLER_H
#define	_CUDARTHANDLER_H

#include <iostream>
#include <map>
#include <string>
#include <cstdio>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cuda_runtime_api.h>

#include "Handler.h"
#include "Result.h"

/**						
 * CudaRtHandler is used by Backend's Process(es) for storing and retrieving
 * device related data and functions. 
 * CudaRtHandler has also the method Execute() that is responsible to execute a
 * named CUDA Runtime routine unmarshalling the input parameters from the
 * provided Buffer.
 */
class CudaRtHandler : public Handler {
public:
    CudaRtHandler();
    virtual ~CudaRtHandler();
    bool CanExecute(std::string routine);
    Result * Execute(std::string routine, Buffer * input_buffer);

    void RegisterFatBinary(std::string & handler, void **fatCubinHandle);
    void RegisterFatBinary(const char * handler, void **fatCubinHandle);
    void **GetFatBinary(std::string & handler);
    void **GetFatBinary(const char * handler);
    void UnregisterFatBinary(std::string & handler);
    void UnregisterFatBinary(const char * handler);

    void RegisterDeviceFunction(std::string & handler, std::string & function);
    void RegisterDeviceFunction(const char * handler, const char * function);
    const char *GetDeviceFunction(std::string & handler);
    const char *GetDeviceFunction(const char * handler);

    void RegisterVar(std::string & handler, std::string &deviceName);
    void RegisterVar(const char *handler, const char *deviceName);
    const char *GetVar(std::string & handler);
    const char *GetVar(const char *handler);

    void RegisterTexture(std::string & handler, textureReference *texref);
    void RegisterTexture(const char *handler, textureReference *texref);
    textureReference *GetTexture(std::string & handler);
    textureReference *GetTexture(const char *handler);
    const char *GetTextureHandler(textureReference *texref);

    const char *GetSymbol(Buffer * in);

private:
    void Initialize();
    typedef Result * (*CudaRoutineHandler)(CudaRtHandler *, Buffer *);
    static std::map<std::string, CudaRoutineHandler> * mspHandlers;
    std::map<std::string, void **> * mpFatBinary;
    std::map<std::string, std::string> * mpDeviceFunction;
    std::map<std::string, std::string> * mpVar;
    std::map<std::string, textureReference *> * mpTexture;
    void *mpShm;
    int mShmFd;
};

#define CUDA_ROUTINE_HANDLER(name) Result * handle##name(CudaRtHandler * pThis, Buffer * input_buffer)
#define CUDA_ROUTINE_HANDLER_PAIR(name) make_pair("cuda" #name, handle##name)

/* CudaRtHandler_device */
CUDA_ROUTINE_HANDLER(ChooseDevice);
CUDA_ROUTINE_HANDLER(GetDevice);
CUDA_ROUTINE_HANDLER(GetDeviceCount);
CUDA_ROUTINE_HANDLER(GetDeviceProperties);
CUDA_ROUTINE_HANDLER(SetDevice);
CUDA_ROUTINE_HANDLER(SetDeviceFlags);
CUDA_ROUTINE_HANDLER(SetValidDevices);

/* CudaRtHandler_error */
CUDA_ROUTINE_HANDLER(GetErrorString);
CUDA_ROUTINE_HANDLER(GetLastError);

/* CudaRtHandler_event */
CUDA_ROUTINE_HANDLER(EventCreate);
CUDA_ROUTINE_HANDLER(EventCreateWithFlags);
CUDA_ROUTINE_HANDLER(EventDestroy);
CUDA_ROUTINE_HANDLER(EventElapsedTime);
CUDA_ROUTINE_HANDLER(EventQuery);
CUDA_ROUTINE_HANDLER(EventRecord);
CUDA_ROUTINE_HANDLER(EventSynchronize);

/* CudaRtHandler_execution */
CUDA_ROUTINE_HANDLER(ConfigureCall);
CUDA_ROUTINE_HANDLER(FuncGetAttributes);
CUDA_ROUTINE_HANDLER(Launch);
CUDA_ROUTINE_HANDLER(SetDoubleForDevice);
CUDA_ROUTINE_HANDLER(SetDoubleForHost);
CUDA_ROUTINE_HANDLER(SetupArgument);

/* CudaRtHandler_internal */
CUDA_ROUTINE_HANDLER(RegisterFatBinary);
CUDA_ROUTINE_HANDLER(UnregisterFatBinary);
CUDA_ROUTINE_HANDLER(RegisterFunction);
CUDA_ROUTINE_HANDLER(RegisterVar);
CUDA_ROUTINE_HANDLER(RegisterSharedVar);
CUDA_ROUTINE_HANDLER(RegisterShared);
CUDA_ROUTINE_HANDLER(RegisterTexture);
CUDA_ROUTINE_HANDLER(RegisterSharedMemory);
CUDA_ROUTINE_HANDLER(RequestSharedMemory);

/* CudaRtHandler_memory */
CUDA_ROUTINE_HANDLER(Free);
CUDA_ROUTINE_HANDLER(FreeArray);
CUDA_ROUTINE_HANDLER(GetSymbolAddress);
CUDA_ROUTINE_HANDLER(GetSymbolSize);
CUDA_ROUTINE_HANDLER(Malloc);
CUDA_ROUTINE_HANDLER(MallocArray);
CUDA_ROUTINE_HANDLER(MallocPitch);
CUDA_ROUTINE_HANDLER(Memcpy);
CUDA_ROUTINE_HANDLER(MemcpyAsync);
CUDA_ROUTINE_HANDLER(MemcpyFromSymbol);
CUDA_ROUTINE_HANDLER(MemcpyToArray);
CUDA_ROUTINE_HANDLER(MemcpyToSymbol);
CUDA_ROUTINE_HANDLER(Memset);

/* CudaRtHandler_stream */
CUDA_ROUTINE_HANDLER(StreamCreate);
CUDA_ROUTINE_HANDLER(StreamDestroy);
CUDA_ROUTINE_HANDLER(StreamQuery);
CUDA_ROUTINE_HANDLER(StreamSynchronize);

/* CudaRtHandler_texture */
CUDA_ROUTINE_HANDLER(BindTexture);
CUDA_ROUTINE_HANDLER(BindTexture2D);
CUDA_ROUTINE_HANDLER(BindTextureToArray);
CUDA_ROUTINE_HANDLER(GetChannelDesc);
CUDA_ROUTINE_HANDLER(GetTextureAlignmentOffset);
CUDA_ROUTINE_HANDLER(GetTextureReference);
CUDA_ROUTINE_HANDLER(UnbindTexture);

/* CudaRtHandler_thread */
CUDA_ROUTINE_HANDLER(ThreadExit);
CUDA_ROUTINE_HANDLER(ThreadSynchronize);

/* CudaRtHandler_version */
CUDA_ROUTINE_HANDLER(DriverGetVersion);
CUDA_ROUTINE_HANDLER(RuntimeGetVersion);

#endif	/* _CUDARTHANDLER_H */

