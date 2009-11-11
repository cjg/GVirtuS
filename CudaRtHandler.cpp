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
    mpDeviceMemory = new map<string, MemoryEntry *>();
    mpFatBinary = new map<string, void **>();
    mpDeviceFunction = new map<string, string>();
    mpVar = new map<string, string>();
    mpTexture = new map<string, textureReference *>();
    Initialize();
}

CudaRtHandler::~CudaRtHandler() {

}

void CudaRtHandler::RegisterDevicePointer(std::string& handler, void* devPtr,
        size_t size) {
    map<string, MemoryEntry *>::iterator it = mpDeviceMemory->find(handler);
    if (it != mpDeviceMemory->end()) {
        /* FIXME: think about freeing memory */
        mpDeviceMemory->erase(it);
    }
    unsigned addr = strtoul(handler.c_str(), NULL, 16);
    mpDeviceMemory->insert(make_pair(handler, new MemoryEntry((void *) addr,
            devPtr, size)));
    cout << "Registered DevicePointer " << devPtr << " with handler " << handler
            << " and a size of " << size << endl;
}

void CudaRtHandler::RegisterDevicePointer(const char* handler, void* devPtr,
        size_t size) {
    string tmp(handler);
    RegisterDevicePointer(tmp, devPtr, size);
}

void * CudaRtHandler::GetDevicePointer(string & handler) {
    if(handler.compare("(nil)") == 0)
        return NULL;
    map<string, MemoryEntry *>::iterator it = mpDeviceMemory->find(handler);
    if (it == mpDeviceMemory->end()) {
        void *hostPtr = (void *) strtoul(handler.c_str(), NULL, 16);
        for(it = mpDeviceMemory->begin(); it != mpDeviceMemory->end(); it++) {
            void *devPtr = it->second->Get(hostPtr);
            if(devPtr != NULL)
                return devPtr;
        }
        throw "Device Pointer '" + handler + "' not found";
    }
    return it->second->Get();
}

void * CudaRtHandler::GetDevicePointer(const char * handler) {
    string tmp(handler);
    return GetDevicePointer(tmp);
}

void CudaRtHandler::UnregisterDevicePointer(std::string& handler) {
    map<string, MemoryEntry *>::iterator it = mpDeviceMemory->find(handler);
    if (it == mpDeviceMemory->end()) {
        /* FIXME: think about throwing an exception */
        return;
    }
    /* FIXME: think about freeing memory */
    cout << "Unregistered DevicePointer " << it->second << " with handler " << handler << endl;
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

void CudaRtHandler::UnregisterFatBinary(std::string& handler) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it == mpFatBinary->end())
        return;
    /* FIXME: think about freeing memory */
    cout << "Unregistered FatBinary " << it->second << " with handler "
            << handler << endl;
    mpFatBinary->erase(it);
}

void CudaRtHandler::UnregisterFatBinary(const char * handler) {
    string tmp(handler);
    UnregisterFatBinary(tmp);
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

void CudaRtHandler::RegisterVar(string & handler, string & symbol) {
    mpVar->insert(make_pair(handler, symbol));
    cout << "Registered Var " << symbol<< " with handler " << handler << endl;
}

void CudaRtHandler::RegisterVar(const char* handler, const char* symbol) {
    string tmp1(handler);
    string tmp2(symbol);
    RegisterVar(tmp1, tmp2);
}

const char *CudaRtHandler::GetVar(string & handler) {
    map<string, string>::iterator it = mpVar->find(handler);
    if(it == mpVar->end())
        return NULL;
    return it->second.c_str();
}

const char * CudaRtHandler::GetVar(const char* handler) {
    string tmp(handler);
    return GetVar(tmp);
}

void CudaRtHandler::RegisterTexture(string& handler, textureReference* texref) {
    mpTexture->insert(make_pair(handler, texref));
    cout << "Registered Texture " << texref << " with handler " << handler
            << endl;
}

void CudaRtHandler::RegisterTexture(const char* handler,
        textureReference* texref) {
    string tmp(handler);
    RegisterTexture(tmp, texref);
}

textureReference *CudaRtHandler::GetTexture(string & handler) {
    map<string, textureReference *>::iterator it = mpTexture->find(handler);
    if(it == mpTexture->end())
        return NULL;
    return it->second;
}

textureReference * CudaRtHandler::GetTexture(const char* handler) {
    string tmp(handler);
    return GetTexture(tmp);
}

void CudaRtHandler::Initialize() {
    if(mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudaRtHandler::CudaRoutineHandler>();

    /* CudaRtHandler_device */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ChooseDevice));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDevice));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceCount));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetDeviceProperties));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDevice));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetDeviceFlags));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetValidDevices));

    /* CudaRtHandler_error */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetErrorString));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetLastError));

    /* CudaRtHandler_event */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventCreate));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventCreateWithFlags));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventDestroy));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventElapsedTime));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventQuery));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventRecord));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(EventSynchronize));

    /* CudaRtHandler_execution */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ConfigureCall));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FuncGetAttributes));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Launch));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(SetupArgument));

    /* CudaRtHandler_internal */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFatBinary));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(UnregisterFatBinary));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterFunction));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterVar));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterShared));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RegisterTexture));

    /* CudaRtHandler_memory */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Free));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(FreeArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Malloc));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MallocArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memcpy));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyAsync));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyFromSymbol));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyToArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(MemcpyToSymbol));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(Memset));

    /* CudaRtHandler_stream */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamCreate));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamDestroy));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamQuery));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(StreamSynchronize));

    /* CudaRtHandler_texture */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(BindTextureToArray));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(GetChannelDesc));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(UnbindTexture));

    /* CudaRtHandler_thread */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ThreadExit));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(ThreadSynchronize));

    /* CudaRtHandler_version */
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(DriverGetVersion));
    mspHandlers->insert(CUDA_ROUTINE_HANDLER_PAIR(RuntimeGetVersion));
}
