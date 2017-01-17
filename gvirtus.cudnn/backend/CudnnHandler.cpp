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
 */

#include "CublasHandler.h"

#include <cstring>

#include <dlfcn.h>

using namespace std;

map<string, CudnnHandler::CudnnRoutineHandler> *CudnnHandler::mspHandlers = NULL;

extern "C" int HandlerInit() {
    return 0;
}
extern "C" Handler *GetHandler() {
    return new OpenclHandler();
}

CudnnHandler::CudnnHandler() {
    mpMapObject = new map<string, string> ();
    pointers = (void **)malloc(sizeof(void*));
    mpMapObject = new map<string, string>();
    nPointers = 1;

    Initialize();
}

CudnnHandler::~CudnnHandler() {

}
bool CublasHandler::CanExecute(std::string routine) {
    map<string, CudnnHandler::CudnnRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end()){
        return false;
        cout<<"false"<<endl;
    }
    return true;
    cout<<"true"<<endl;
}

Result * CudnnHandler::Execute(std::string routine, Buffer * input_buffer) {
    map<string, CudnnHandler::CudnnRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    return it->second(this, input_buffer);
}

void* CudnnHandler::RegisterPointer(void* pointer,size_t bytes){
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

    void CudnnHandler::RegisterMapObject(char * key,char * value){

        mpMapObject->insert(make_pair(key, value));

    }
    char * CudnnHandler::GetMapObject(char * key){
        for (map<string, string>::iterator it = mpMapObject->begin();
            it != mpMapObject->end(); it++)
        if (it->first == key)
            return (char *)(it->second.c_str());
    return NULL;
    }

void CudnnHandler::Initialize() {
    if (mspHandlers != NULL)
        return;
    pointers = (void **)malloc(sizeof(void*));
    mpMapObject = new map<string, string>();
    nPointers = 1;
    mspHandlers = new map<string, CudnnHandler::CudnnRoutineHandler> ();

    /* CublasHandler Query Platform Info */
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(cublasCreate));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(cublasSetStream));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(cublasGetStream));
    mspHandlers->insert(CUDNN_ROUTINE_HANDLER_PAIR(cublasDestroy));
}
