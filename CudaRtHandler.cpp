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

static cudaError_t GetDeviceCount(CudaRtHandler *pThis, char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size) {
    /* cudaError_t cudaGetDeviceCount(int *count) */

    int *count = (int *) in_buffer;

    cudaError_t result = cudaGetDeviceCount(count);

    if (result == cudaSuccess) {
        *out_buffer_size = sizeof (int);
        *out_buffer = new char[*out_buffer_size];
        memmove(*out_buffer, count, sizeof (int));
    }

    return result;
}

static cudaError_t GetDeviceProperties(CudaRtHandler *pThis, char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size) {
    /* cudaError_t cudaGetDeviceCount(struct cudaDeviceProp *prop,
       int device) */
    struct cudaDeviceProp *prop = (struct cudaDeviceProp *) in_buffer;
    int device = *((int *) (in_buffer + sizeof (struct cudaDeviceProp)));

    cudaError_t result = cudaGetDeviceProperties(prop, device);

    if (result == cudaSuccess) {
        *out_buffer_size = sizeof (struct cudaDeviceProp);
        *out_buffer = new char[*out_buffer_size];
        memmove(*out_buffer, prop, sizeof (struct cudaDeviceProp));
    }

    return result;
}

static cudaError_t Free(CudaRtHandler *pThis, char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size) {
    /* cudaError_t cudaFree(void *devPtr) */
    void *devPtr = pThis->GetDevicePointer(in_buffer);
    cudaError_t result = cudaFree(devPtr);
    pThis->UnregisterDevicePointer(in_buffer);

    if(result == cudaSuccess) {
        *out_buffer_size = 0;
        *out_buffer = NULL;
    }

    return result;
}

static cudaError_t Malloc(CudaRtHandler *pThis, char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size) {
    /* cudaError_t cudaMalloc(void **devPtr, size_t size) */
    void *devPtr = NULL;
    size_t size = *((size_t *) (in_buffer + sizeof (void *) * 2 + 3));

    cudaError_t result = cudaMalloc(&devPtr, size);
    pThis->RegisterDevicePointer(in_buffer, devPtr);


    if (result == cudaSuccess) {
        *out_buffer_size = 0;
        *out_buffer = NULL;
    }

    return result;
}

static cudaError_t Memcpy(CudaRtHandler *pThis, char *in_buffer,
        size_t in_buffer_size, char **out_buffer, size_t *out_buffer_size) {
    /* cudaError_t cudaError_t cudaMemcpy(void *dst, const void *src,
        size_t count, cudaMemcpyKind kind) */
    void *dst = NULL;
    void *src = NULL;
    size_t count = *((size_t *) (in_buffer + in_buffer_size
            - sizeof(size_t) - sizeof(cudaMemcpyKind)));
    cudaMemcpyKind kind = *((cudaMemcpyKind *) (in_buffer + in_buffer_size
            - sizeof(cudaMemcpyKind)));
    cudaError_t result;

    switch(kind) {
        case cudaMemcpyHostToHost:
            // This should nevere happer
            break;
        case cudaMemcpyHostToDevice:
            dst = pThis->GetDevicePointer(in_buffer);
            src = in_buffer + CudaUtil::MarshaledDevicePointerSize;
            result = cudaMemcpy(dst, src, count, kind);
            if(result == cudaSuccess) {
                *out_buffer_size = 0;
                *out_buffer = NULL;
            }
            break;
        case cudaMemcpyDeviceToHost:
            *out_buffer_size = count;
            *out_buffer = new char[*out_buffer_size];
            dst = *out_buffer;
            /* adding +1 for fake host pointer */
            src = pThis->GetDevicePointer(in_buffer + 1);
            result = cudaMemcpy(dst, src, count, kind);
            break;
        case cudaMemcpyDeviceToDevice:
            dst = pThis->GetDevicePointer(in_buffer);
            src = pThis->GetDevicePointer(in_buffer
                    + CudaUtil::MarshaledDevicePointerSize);
            result = cudaMemcpy(dst, src, count, kind);
            if(result == cudaSuccess) {
                *out_buffer_size = 0;
                *out_buffer = NULL;
            }
            break;
    }
    return result;
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

cudaError_t CudaRtHandler::Execute(std::string routine, char* in_buffer,
    size_t in_buffer_size, char** out_buffer, size_t* out_buffer_size) {
    map<string, CudaRtHandler::CudaRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if(it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    return it->second(this, in_buffer, in_buffer_size, out_buffer,
        out_buffer_size);
}

void CudaRtHandler::Initialize() {
    if(mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudaRtHandler::CudaRoutineHandler>();

    /* Thread Management */
    mspHandlers->insert(make_pair("cudaThreadSynchronize", ThreadSynchronize));

    /* Error Handling */
    mspHandlers->insert(make_pair("cudaGetLastError", GetLastError));

    /* Memory Management */
    mspHandlers->insert(make_pair("cudaFree", Free));
    mspHandlers->insert(make_pair("cudaMalloc", Malloc));
    mspHandlers->insert(make_pair("cudaMemcpy", Memcpy));

    mspHandlers->insert(make_pair("cudaGetDeviceCount", GetDeviceCount));
    mspHandlers->insert(make_pair("cudaGetDeviceProperties", GetDeviceProperties));

}