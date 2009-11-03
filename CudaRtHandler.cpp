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
    mpFatBinary = new map<string, void **>();
    mpDeviceFunction = new map<string, string>();
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

void CudaRtHandler::RegisterFatBinary(std::string& handler, void ** fatCubinHandle) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it != mpFatBinary->end()) {
        /* FIXME: think about freeing memory */
        mpFatBinary->erase(it);
    }
    mpFatBinary->insert(make_pair(handler, fatCubinHandle));
    cout << "Registered FatBinary " << fatCubinHandle << " with handler " << handler << endl;
}

void CudaRtHandler::RegisterFatBinary(const char* handler, void ** fatCubinHandle) {
    string tmp(handler);
    RegisterFatBinary(tmp, fatCubinHandle);
}

void ** CudaRtHandler::GetFatBinary(string & handler) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it == mpFatBinary->end())
        throw "Fat Binary '" + handler + "' not found";
    return it->second;
}

void ** CudaRtHandler::GetFatBinary(const char * handler) {
    string tmp(handler);
    return GetFatBinary(tmp);
}

void CudaRtHandler::RegisterDeviceFunction(std::string & handler, std::string & function) {
    map<string, string>::iterator it = mpDeviceFunction->find(handler);
    if(it != mpDeviceFunction->end())
        mpDeviceFunction->erase(it);
    mpDeviceFunction->insert(make_pair(handler, function));
    cout << "Registered DeviceFunction " << function << " with handler " << handler << endl;
}

void CudaRtHandler::RegisterDeviceFunction(const char * handler, const char * function) {
    string tmp1(handler);
    string tmp2(function);
    RegisterDeviceFunction(tmp1, tmp2);
}

const char *CudaRtHandler::GetDeviceFunction(std::string & handler) {
    map<string, string>::iterator it = mpDeviceFunction->find(handler);
    if(it == mpDeviceFunction->end())
        throw "Device Function '" +  handler + "' not fount";
    return it->second.c_str();
}

const char *CudaRtHandler::GetDeviceFunction(const char * handler) {
    string tmp(handler);
    return GetDeviceFunction(tmp);
}


void CudaRtHandler::Initialize() {
    if(mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudaRtHandler::CudaRoutineHandler>();

    /* CudaRtHandler_device */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceCount));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceProperties));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDevice));

    /* CudaRtHandler_error */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetErrorString));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetLastError));

    /* CudaRtHandler_execution */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ConfigureCall));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Launch));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetupArgument));

    /* CudaRtHandler_internal */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFatBinary));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFunction));

    /* CudaRtHandler_memory */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Free));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Malloc));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy));
}