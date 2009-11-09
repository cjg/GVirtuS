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
private:
    void Initialize();
    typedef Result * (*CudaRoutineHandler)(CudaRtHandler *, Buffer *);
    static std::map<std::string, CudaRoutineHandler> * mspHandlers;
    std::map<std::string, void *> * mpDeviceMemory;
    std::map<std::string, void **> * mpFatBinary;
    std::map<std::string, std::string> * mpDeviceFunction;
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

/* CudaRtHandler_execution */
CUDA_ROUTINE_HANDLER(ConfigureCall);
CUDA_ROUTINE_HANDLER(Launch);
CUDA_ROUTINE_HANDLER(SetupArgument);

/* CudaRtHandler_internal */
CUDA_ROUTINE_HANDLER(RegisterFatBinary);
CUDA_ROUTINE_HANDLER(UnregisterFatBinary);
CUDA_ROUTINE_HANDLER(RegisterFunction);

/* CudaRtHandler_memory */
CUDA_ROUTINE_HANDLER(Free);
CUDA_ROUTINE_HANDLER(Malloc);
CUDA_ROUTINE_HANDLER(Memcpy);
CUDA_ROUTINE_HANDLER(Memset);

/* CudaRtHandler_thread */
CUDA_ROUTINE_HANDLER(ThreadExit);
CUDA_ROUTINE_HANDLER(ThreadSynchronize);

/* CudaRtHandler_version */
CUDA_ROUTINE_HANDLER(DriverGetVersion);
CUDA_ROUTINE_HANDLER(RuntimeGetVersion);

#endif	/* _CUDARTHANDLER_H */

