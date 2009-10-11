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

class CudaRtHandler {
public:
    CudaRtHandler();
    virtual ~CudaRtHandler();
    cudaError_t Execute(std::string routine, char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size);
    void RegisterDevicePointer(std::string & handler, void *devPtr);
    void RegisterDevicePointer(const char * handler, void *devPtr);
    void *GetDevicePointer(std::string & handler);
    void *GetDevicePointer(const char * handler);
    void UnregisterDevicePointer(std::string & handler);
    void UnregisterDevicePointer(const char * handler);
private:
    void Initialize();
    typedef cudaError_t (*CudaRoutineHandler)(CudaRtHandler *, char *, size_t, char **, size_t *);
    static std::map<std::string, CudaRoutineHandler> * mspHandlers;
    std::map<std::string, void *> * mpDeviceMemory;
};

#endif	/* _CUDARTHANDLER_H */

