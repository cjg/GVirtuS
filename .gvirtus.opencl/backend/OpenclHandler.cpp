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
 * @file   Backend.cpp
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 *
 * @brief
 *
 *
 */

#include "OpenclHandler.h"

#include <cstring>

//#include <cuda_runtime_api.h>

//#include "CudaUtil.h"

#include <dlfcn.h>

using namespace std;

map<string, OpenclHandler::OpenclRoutineHandler> *OpenclHandler::mspHandlers = NULL;

extern "C" int HandlerInit() {
    return 0;
}
extern "C" Handler *GetHandler() {
    return new OpenclHandler();
}

OpenclHandler::OpenclHandler() {
    mpMapObject = new map<string, string> ();
    pointers = (void **)malloc(sizeof(void*));
    mpMapObject = new map<string, string>();
    nPointers = 1;

    Initialize();
}

OpenclHandler::~OpenclHandler() {

}
bool OpenclHandler::CanExecute(std::string routine) {
    map<string, OpenclHandler::OpenclRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end()){
        return false;
        cout<<"false"<<endl;
    }
    return true;
    cout<<"true"<<endl;
}

Result * OpenclHandler::Execute(std::string routine, Buffer * input_buffer) {
    map<string, OpenclHandler::OpenclRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    return it->second(this, input_buffer);
}

void* OpenclHandler::RegisterPointer(void* pointer,size_t bytes){
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

    void OpenclHandler::RegisterMapObject(char * key,char * value){

        mpMapObject->insert(make_pair(key, value));

    }
    char * OpenclHandler::GetMapObject(char * key){
        for (map<string, string>::iterator it = mpMapObject->begin();
            it != mpMapObject->end(); it++)
        if (it->first == key)
            return (char *)(it->second.c_str());
    return NULL;
    }

void OpenclHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    pointers = (void **)malloc(sizeof(void*));
    mpMapObject = new map<string, string>();
    nPointers = 1;
    mspHandlers = new map<string, OpenclHandler::OpenclRoutineHandler> ();

    /* OclHandler Query Platform Info */
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(GetPlatformIDs));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(GetDeviceIDs));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(CreateContext));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(CreateCommandQueue));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(CreateProgramWithSource));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(BuildProgram));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(CreateBuffer));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(CreateKernel));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(SetKernelArg));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(EnqueueWriteBuffer));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(EnqueueNDRangeKernel));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(EnqueueReadBuffer));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(Finish));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(GetPlatformInfo));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(GetContextInfo));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(Flush));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(EnqueueCopyBuffer));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(ReleaseMemObject));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(ReleaseKernel));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(ReleaseContext));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(ReleaseCommandQueue));
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(ReleaseProgram));
    //mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(ReleaseEvent));
    //mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(WaitForEvents));

}












/*
extern "C" int HandlerInit() {
    return 0;
}

extern "C" Handler *GetHandler() {
    return new OpenclHandler();
}

OpenclHandler::OpenclHandler() {
    mpFatBinary = new map<string, void **>();
    mpDeviceFunction = new map<string, string > ();
    mpVar = new map<string, string > ();
    //mpTexture = new map<string, textureReference *>();
    Initialize();
}

OpenclHandler::~OpenclHandler() {

}

bool OpenclHandler::CanExecute(std::string routine) {
    map<string, OpenclHandler::OpenclRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        return false;
    return true;
}

Result * OpenclHandler::Execute(std::string routine, Buffer * input_buffer) {
    map<string, OpenclHandler::OpenclRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    return it->second(this, input_buffer);
}

void OpenclHandler::RegisterFatBinary(std::string& handler, void ** fatCubinHandle) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it != mpFatBinary->end()) {
        mpFatBinary->erase(it);
    }
    mpFatBinary->insert(make_pair(handler, fatCubinHandle));
    cout << "Registered FatBinary " << fatCubinHandle << " with handler " << handler << endl;
}

void OpenclHandler::RegisterFatBinary(const char* handler, void ** fatCubinHandle) {
    string tmp(handler);
    RegisterFatBinary(tmp, fatCubinHandle);
}

void ** OpenclHandler::GetFatBinary(string & handler) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it == mpFatBinary->end())
        throw "Fat Binary '" + handler + "' not found";
    return it->second;
}

void ** OpenclHandler::GetFatBinary(const char * handler) {
    string tmp(handler);
    return GetFatBinary(tmp);
}

void OpenclHandler::UnregisterFatBinary(std::string& handler) {
    map<string, void **>::iterator it = mpFatBinary->find(handler);
    if (it == mpFatBinary->end())
        return;
    // FIXME: think about freeing memory
    cout << "Unregistered FatBinary " << it->second << " with handler "
            << handler << endl;
    mpFatBinary->erase(it);
}

void OpenclHandler::UnregisterFatBinary(const char * handler) {
    string tmp(handler);
    UnregisterFatBinary(tmp);
}

void OpenclHandler::RegisterDeviceFunction(std::string & handler, std::string & function) {
    map<string, string>::iterator it = mpDeviceFunction->find(handler);
    if (it != mpDeviceFunction->end())
        mpDeviceFunction->erase(it);
    mpDeviceFunction->insert(make_pair(handler, function));
    cout << "Registered DeviceFunction " << function << " with handler " << handler << endl;
}

void OpenclHandler::RegisterDeviceFunction(const char * handler, const char * function) {
    string tmp1(handler);
    string tmp2(function);
    RegisterDeviceFunction(tmp1, tmp2);
}

const char *OpenclHandler::GetDeviceFunction(std::string & handler) {
    map<string, string>::iterator it = mpDeviceFunction->find(handler);
    if (it == mpDeviceFunction->end())
        throw "Device Function '" + handler + "' not fount";
    return it->second.c_str();
}

const char *OpenclHandler::GetDeviceFunction(const char * handler) {
    string tmp(handler);
    return GetDeviceFunction(tmp);
}

void OpenclHandler::RegisterVar(string & handler, string & symbol) {
    mpVar->insert(make_pair(handler, symbol));
    cout << "Registered Var " << symbol << " with handler " << handler << endl;
}

void OpenclHandler::RegisterVar(const char* handler, const char* symbol) {
    string tmp1(handler);
    string tmp2(symbol);
    RegisterVar(tmp1, tmp2);
}

const char *OpenclHandler::GetVar(string & handler) {
    map<string, string>::iterator it = mpVar->find(handler);
    if (it == mpVar->end()) 
        return NULL;
    return it->second.c_str();
}

const char * OpenclHandler::GetVar(const char* handler) {
    string tmp(handler);
    return GetVar(tmp);
}
*/

/*
void OpenclHandler::RegisterTexture(string& handler, textureReference* texref) {
    mpTexture->insert(make_pair(handler, texref));
    cout << "Registered Texture " << texref << " with handler " << handler
            << endl;
}

void OpenclHandler::RegisterTexture(const char* handler,
        textureReference* texref) {
    string tmp(handler);
    RegisterTexture(tmp, texref);
}

textureReference *OpenclHandler::GetTexture(string & handler) {
    map<string, textureReference *>::iterator it = mpTexture->find(handler);
    if (it == mpTexture->end())
        return NULL;
    return it->second;
}

textureReference * OpenclHandler::GetTexture(const char* handler) {
    string tmp(handler);
    return GetTexture(tmp);
}

const char *OpenclHandler::GetTextureHandler(textureReference* texref) {
    for (map<string, textureReference *>::iterator it = mpTexture->begin();
            it != mpTexture->end(); it++)
        if (it->second == texref)
            return it->first.c_str();
    return NULL;
}

const char *OpenclHandler::GetSymbol(Buffer* in) {
    char *symbol_handler = in->AssignString();
    char *symbol = in->AssignString();
    char *our_symbol = const_cast<char *> (GetVar(symbol_handler));
    if (our_symbol != NULL)
        symbol = const_cast<char *> (our_symbol);
    return symbol;
}

void OpenclHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, OpenclHandler::OpenclRoutineHandler > ();

    // OpenclHandler_Platform
    mspHandlers->insert(OPENCL_ROUTINE_HANDLER_PAIR(GetPlatformIDs));
   
}*/
