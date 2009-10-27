/* 
 * File:   CudaRtHandler.h
 * Author: cjg
 *
 * Created on October 10, 2009, 10:51 PM
 */

#ifndef _CUDARTHANDLER_H
#define	_CUDARTHANDLER_H

#include <iostream>
#include <map>
#include <string>
#include <host_defines.h>
#include <builtin_types.h>
#include <driver_types.h>
#include "Result.h"

class CudaRtHandler {
public:
    CudaRtHandler();
    virtual ~CudaRtHandler();
    Result * Execute(std::string routine, Buffer * input_buffer);
    void RegisterDevicePointer(std::string & handler, void *devPtr);
    void RegisterDevicePointer(const char * handler, void *devPtr);
    void *GetDevicePointer(std::string & handler);
    void *GetDevicePointer(const char * handler);
    void UnregisterDevicePointer(std::string & handler);
    void UnregisterDevicePointer(const char * handler);
private:
    void Initialize();
    typedef Result * (*CudaRoutineHandler)(CudaRtHandler *, Buffer *);
    static std::map<std::string, CudaRoutineHandler> * mspHandlers;
    std::map<std::string, void *> * mpDeviceMemory;
};

#define CUDA_ROUTINE_HANDLER(name) Result * handle##name(CudaRtHandler * pThis, Buffer * input_buffer)

/* CudaRtHandler_device */
CUDA_ROUTINE_HANDLER(GetDeviceCount);
CUDA_ROUTINE_HANDLER(GetDeviceProperties);

/* CudaRtHandler_memory */
CUDA_ROUTINE_HANDLER(Free);
CUDA_ROUTINE_HANDLER(Malloc);
CUDA_ROUTINE_HANDLER(Memcpy);

#endif	/* _CUDARTHANDLER_H */

