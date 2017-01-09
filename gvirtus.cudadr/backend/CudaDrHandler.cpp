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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */
//#define DEBUG
#include <host_defines.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <GL/gl.h>
#include <cuda_runtime_api.h>
#include <cstring>
#include "CudaUtil.h"
#include "CudaDrHandler.h"
#include <cuda.h>

using namespace std;
using namespace log4cplus;

map<string, CudaDrHandler::CudaDriverHandler> *CudaDrHandler::mspHandlers = NULL;

extern "C" int HandlerInit() {
    return 0;
}

extern "C" Handler *GetHandler() {
    return new CudaDrHandler();
}

CudaDrHandler::CudaDrHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CudaDrHandler"));
    mpFatBinary = new map<string, void **>();
    mpDeviceFunction = new map<string, string > ();
    mpVar = new map<string, string > ();
    mpTexture = new map<string, textureReference *>();
    Initialize();
}

CudaDrHandler::~CudaDrHandler() {

}

bool CudaDrHandler::CanExecute(std::string routine) {
    map<string, CudaDrHandler::CudaDriverHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        return false;
    return true;
}

Result * CudaDrHandler::Execute(std::string routine, Buffer * input_buffer) {
    map<string, CudaDrHandler::CudaDriverHandler>::iterator it;
//#ifdef DEBUG
//    std::cout<<"Called "<<routine<<std::endl;
//#endif    
    LOG4CPLUS_DEBUG(logger,"Called " << routine);

    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    return it->second(this, input_buffer);
}

void CudaDrHandler::RegisterFatBinary(std::string& handler, void ** fatCubinHandle) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it != mpFatBinary->end()) {
        mpFatBinary->erase(it);
    }
    mpFatBinary->insert(make_pair(handler, fatCubinHandle));
//#ifdef DEBUG
//    cout << "Registered FatBinary " << fatCubinHandle << " with handler " << handler << endl;
//#endif 
    LOG4CPLUS_DEBUG(logger, "Registered FatBinary " << fatCubinHandle << " with handler " << handler);
}

void CudaDrHandler::RegisterFatBinary(const char* handler, void ** fatCubinHandle) {
    string tmp(handler);
    RegisterFatBinary(tmp, fatCubinHandle);
}

void ** CudaDrHandler::GetFatBinary(string & handler) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it == mpFatBinary->end())
        throw "Fat Binary '" + handler + "' not found";
    return it->second;
}

void ** CudaDrHandler::GetFatBinary(const char * handler) {
    string tmp(handler);
    return GetFatBinary(tmp);
}

void CudaDrHandler::UnregisterFatBinary(std::string& handler) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it == mpFatBinary->end())
        return;
    /* FIXME: think about freeing memory */
//#ifdef DEBUG
//    cout << "Unregistered FatBinary " << it->second << " with handler "<< handler << endl;
//#endif
    LOG4CPLUS_DEBUG(logger, "Unregistered FatBinary " << it->second << " with handler "<< handler);
    mpFatBinary->erase(it);
}

void CudaDrHandler::UnregisterFatBinary(const char * handler) {
    string tmp(handler);
    UnregisterFatBinary(tmp);
}

void CudaDrHandler::RegisterDeviceFunction(std::string & handler, std::string & function) {
    map<string, string>::iterator it = mpDeviceFunction->find(handler);
    if (it != mpDeviceFunction->end())
        mpDeviceFunction->erase(it);
    mpDeviceFunction->insert(make_pair(handler, function));
//#ifdef DEBUG
//    cout << "Registered DeviceFunction " << function << " with handler " << handler << endl;
//#endif
    LOG4CPLUS_DEBUG(logger, "Registered DeviceFunction " << function << " with handler " << handler);
}

void CudaDrHandler::RegisterDeviceFunction(const char * handler, const char * function) {
    string tmp1(handler);
    string tmp2(function);
    RegisterDeviceFunction(tmp1, tmp2);
}

const char *CudaDrHandler::GetDeviceFunction(std::string & handler) {
    map<string, string>::iterator it = mpDeviceFunction->find(handler);
    if (it == mpDeviceFunction->end())
        throw "Device Function '" + handler + "' not found";
    return it->second.c_str();
}

const char *CudaDrHandler::GetDeviceFunction(const char * handler) {
    string tmp(handler);
    return GetDeviceFunction(tmp);
}

void CudaDrHandler::RegisterVar(string & handler, string & symbol) {
    mpVar->insert(make_pair(handler, symbol));
//#ifdef DEBUG
//    cout << "Registered Var " << symbol << " with handler " << handler << endl;
//#endif
    LOG4CPLUS_DEBUG(logger,"Registered Var " << symbol << " with handler " << handler );
}

void CudaDrHandler::RegisterVar(const char* handler, const char* symbol) {
    string tmp1(handler);
    string tmp2(symbol);
    RegisterVar(tmp1, tmp2);
}

const char *CudaDrHandler::GetVar(string & handler) {
    map<string, string>::iterator it = mpVar->find(handler);
    if (it == mpVar->end())
        return NULL;
    return it->second.c_str();
}

const char * CudaDrHandler::GetVar(const char* handler) {
    string tmp(handler);
    return GetVar(tmp);
}

void CudaDrHandler::RegisterTexture(string& handler, textureReference* texref) {
    mpTexture->insert(make_pair(handler, texref));
//#ifdef DEBUG
//    cout << "Registered Texture " << texref << " with handler " << handler<< endl;
//#endif
    LOG4CPLUS_DEBUG(logger,"Registered Texture " << texref << " with handler " << handler);
}

void CudaDrHandler::RegisterTexture(const char* handler,
        textureReference* texref) {
    string tmp(handler);
    RegisterTexture(tmp, texref);
}

textureReference *CudaDrHandler::GetTexture(string & handler) {
    map<string, textureReference *>::iterator it = mpTexture->find(handler);
    if (it == mpTexture->end())
        return NULL;
    return it->second;
}

textureReference * CudaDrHandler::GetTexture(const char* handler) {
    string tmp(handler);
    return GetTexture(tmp);
}

const char *CudaDrHandler::GetTextureHandler(textureReference* texref) {
    for (map<string, textureReference *>::iterator it = mpTexture->begin();
            it != mpTexture->end(); it++)
        if (it->second == texref)
            return it->first.c_str();
    return NULL;
}

const char *CudaDrHandler::GetSymbol(Buffer* in) {
    char *symbol_handler = in->AssignString();
    char *symbol = in->AssignString();
    char *our_symbol = const_cast<char *> (GetVar(symbol_handler));
    if (our_symbol != NULL)
        symbol = const_cast<char *> (our_symbol);
    return symbol;
}

void CudaDrHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudaDrHandler::CudaDriverHandler > ();

    /*CudaDrHAndler_initialization*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(Init));

    /*CudaDrHandler_context*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxCreate));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxAttach));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxDestroy));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxDetach));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxGetDevice));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxPopCurrent));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxPushCurrent));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxSynchronize));

    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxDisablePeerAccess));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxEnablePeerAccess));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceCanAccessPeer));

    /*CudaDrHandler_device*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceComputeCapability));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceGet));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceGetAttribute));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceGetCount));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceGetName));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceGetProperties));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceTotalMem));

    /*CudaDrHandler_execution*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ParamSetSize));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(FuncSetBlockShape));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(LaunchGrid));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(FuncGetAttribute));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(FuncSetSharedSize));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(Launch));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ParamSetf));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ParamSeti));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ParamSetv));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ParamSetTexRef));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(LaunchGridAsync));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(FuncSetCacheConfig));


    /*CudaDrHandler_memory*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(MemFree));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(MemAlloc));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(MemcpyDtoH));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(MemcpyHtoD));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ArrayCreate));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(Memcpy2D));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ArrayDestroy));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(Array3DCreate));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(MemAllocPitch));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(MemGetAddressRange));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(MemGetInfo));




    /*CudaDrHandler_module*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ModuleLoadData));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ModuleGetFunction));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ModuleGetGlobal));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ModuleLoadDataEx));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ModuleLoad));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ModuleLoadFatBinary));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ModuleUnload));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(ModuleGetTexRef));


    /*CudaDrHandler_version*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DriverGetVersion));

    /*CudaDrHandler_stream*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(StreamCreate));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(StreamDestroy));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(StreamQuery));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(StreamSynchronize));

    /*CudaDrHandler_event*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(EventCreate));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(EventDestroy));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(EventElapsedTime));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(EventQuery));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(EventRecord));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(EventSynchronize));

    /*CudaDrHandler_texture*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefSetArray));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefSetAddressMode));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefSetFilterMode));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefSetFlags));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefSetFormat));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefGetAddress));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefGetArray));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefGetFlags));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(TexRefSetAddress));
    
    /*New Cuda 4.0 functions*/
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(LaunchKernel));

    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxDisablePeerAccess));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(CtxEnablePeerAccess));
    mspHandlers->insert(CUDA_DRIVER_HANDLER_PAIR(DeviceCanAccessPeer));


}
