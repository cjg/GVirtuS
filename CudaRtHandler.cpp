/* 
 * File:   CudaRtHandler.cpp
 * Author: cjg
 * 
 * Created on October 10, 2009, 10:51 PM
 */

#include <host_defines.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "CudaUtil.h"
#include "CudaRtHandler.h"

using namespace std;

map<string, CudaRtHandler::CudaRoutineHandler> *CudaRtHandler::mspHandlers = NULL;

CudaRtHandler::CudaRtHandler() {
    mpDeviceMemory = new map<string, void *>();
    Initialize();
}

CudaRtHandler::~CudaRtHandler() {

}

static cudaError_t ThreadSynchronize(CudaRtHandler *pThis, char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size) {
    /* cudaError_t cudaThreadSynchronize(void) */
    cudaError_t result = cudaThreadSynchronize();

    if(result == cudaSuccess) {
        *out_buffer_size = 0;
        *out_buffer = NULL;
    }

    return result;
}

static cudaError_t GetLastError(CudaRtHandler *pThis, char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size) {
    /* cudaError_t cudaThreadSynchronize(void) */
    cudaError_t result = cudaGetLastError();

    if(result == cudaSuccess) {
        *out_buffer_size = 0;
        *out_buffer = NULL;
    }

    return result;
}

void CudaRtHandler::RegisterDevicePointer(std::string& handler, void* devPtr) {
    map<string, void *>::iterator it = mpDeviceMemory->find(handler);
    if (it != mpDeviceMemory->end()) {
        /* FIXME: think about freeing memory */
        mpDeviceMemory->erase(it);
    }
    mpDeviceMemory->insert(make_pair(handler, devPtr));
    cout << "Registered DevicePointer " << devPtr << " with handler " << handler << endl;
}

void CudaRtHandler::RegisterDevicePointer(const char* handler, void* devPtr) {
    string tmp(handler);
    RegisterDevicePointer(tmp, devPtr);
}

void * CudaRtHandler::GetDevicePointer(string & handler) {
    map<string, void *>::iterator it = mpDeviceMemory->find(handler);
    if (it == mpDeviceMemory->end()) 
        throw "Device Pointer '" + handler + "' not found";
    return it->second;
}

void * CudaRtHandler::GetDevicePointer(const char * handler) {
    string tmp(handler);
    return GetDevicePointer(tmp);
}

void CudaRtHandler::UnregisterDevicePointer(std::string& handler) {
    map<string, void *>::iterator it = mpDeviceMemory->find(handler);
    if (it == mpDeviceMemory->end()) {
        /* FIXME: think about throwing an exception */
        return;
    }
    /* FIXME: think about freeing memory */
    cout << "Registered DevicePointer " << it->second << " with handler " << handler << endl;
    mpDeviceMemory->erase(it);
}

void CudaRtHandler::UnregisterDevicePointer(const char* handler) {
    string tmp(handler);
    UnregisterDevicePointer(tmp);
}

Result * CudaRtHandler::Execute(std::string routine, Buffer * input_buffer) {
    map<string, CudaRtHandler::CudaRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if(it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    return it->second(this, input_buffer);
}

void CudaRtHandler::Initialize() {
    if(mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudaRtHandler::CudaRoutineHandler>();

    /* CudaRtHandler_device */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceCount));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceProperties));

    /* CudaRtHandler_device */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Free));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Malloc));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy));
}