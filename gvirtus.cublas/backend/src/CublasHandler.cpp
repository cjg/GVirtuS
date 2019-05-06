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
#include "CublasHandler_Helper.cpp"
#include "CublasHandler_Level1.cpp"
#include "CublasHandler_Level2.cpp"
#include "CublasHandler_Level3.cpp"
#include <cstring>
//#include <map>
#include <bits/stl_map.h>
#include <errno.h>

using namespace std;
using namespace log4cplus;

std::map<string, CublasHandler::CublasRoutineHandler> *CublasHandler::mspHandlers = NULL;

extern "C" int
HandlerInit() {
  return 0;
}
extern "C" Handler *
GetHandler() {
  return new CublasHandler();
}

CublasHandler::CublasHandler() {
  logger = Logger::getInstance(LOG4CPLUS_TEXT("CufftHandler"));
  Initialize();
}

CublasHandler::~CublasHandler() {}

bool
CublasHandler::CanExecute(std::string routine) {
  return mspHandlers->find(routine) != mspHandlers->end();
}

Result *
CublasHandler::Execute(std::string routine, Buffer *input_buffer) {
  LOG4CPLUS_DEBUG(logger, "Called " << routine);
  map<string, CublasHandler::CublasRoutineHandler>::iterator it;
  it = mspHandlers->find(routine);
  if (it == mspHandlers->end())
    throw "No handler for '" + routine + "' found!";
  try {
    return it->second(this, input_buffer);
  } catch (const char *ex) {
    cout << ex << endl;
    cout << strerror(errno) << endl;
  }
  return NULL;
}

/*void* CublasHandler::RegisterPointer(void* pointer,size_t bytes){
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

void CublasHandler::RegisterMapObject(char * key,char * value){

    mpMapObject->insert(make_pair(key, value));

}

char * CublasHandler::GetMapObject(char * key){
    for (map<string, string>::iterator it = mpMapObject->begin();
        it != mpMapObject->end(); it++)
    if (it->first == key)
        return (char *)(it->second.c_str());
    return NULL;
}*/

void
CublasHandler::Initialize() {
  if (mspHandlers != NULL)
    return;
  mspHandlers = new map<string, CublasHandler::CublasRoutineHandler>();

  // pointers = (void **)malloc(sizeof(void*));
  // mpMapObject = new map<string, string>();
  // nPointers = 1;

  /* CublasHandler Helper functions */
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetVersion_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Create_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Destroy_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetVector));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetVector));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetMatrix));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetMatrix));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetStream_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(GetPointerMode_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SetPointerMode_v2));

  /* CublasHandler Level1 functions */
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sdot_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ddot_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cdotu_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cdotc_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zdotu_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zdotc_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sscal_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dscal_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cscal_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csscal_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zscal_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zdscal_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Saxpy_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Daxpy_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Caxpy_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zaxpy_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Scopy_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ccopy_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dcopy_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zcopy_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sswap_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dswap_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cswap_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zswap_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Isamax_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Idamax_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Icamax_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Izamax_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sasum_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dasum_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Scasum_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dzasum_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Srot_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Drot_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Crot_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csrot_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zrot_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zdrot_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Srotg_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Drotg_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Crotg_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zrotg_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Srotm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Drotm_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Srotmg_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Drotmg_v2));

  /* CublasHandler Level2 functions */
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgemv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dgemv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cgemv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zgemv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dgbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgbmv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Strmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtrmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctrmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztrmv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Stbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztbmv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Stpmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtpmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctpmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztpmv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Stpsv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtpsv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctpsv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztpsv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Stbsv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtbsv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctbsv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztbsv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssymv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsymv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csymv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsymv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chemv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhemv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chbmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhbmv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sspmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dspmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chpmv_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhpmv_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sger_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dger_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cgeru_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cgerc_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zgeru_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zgerc_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssyr_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsyr_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csyr_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsyr_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cher_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zher_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sspr_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dspr_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chpr_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhpr_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssyr2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsyr2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csyr2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsyr2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cher2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zher2_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sspr2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dspr2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chpr2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhpr2_v2));
  /* CublasHandler Level3 functions */
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Sgemm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dgemm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cgemm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zgemm_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(SgemmBatched_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(DgemmBatched_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(CgemmBatched_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(ZgemmBatched_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Snrm2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dnrm2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Scnrm2_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dznrm2_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssyrk_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsyrk_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csyrk_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsyrk_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cherk_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zherk_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssyr2k_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsyr2k_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csyr2k_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsyr2k_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Cher2k_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zher2k_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ssymm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dsymm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Csymm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zsymm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Chemm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Zhemm_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Strsm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtrsm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctrsm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztrsm_v2));

  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Strmm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Dtrmm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ctrmm_v2));
  mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(Ztrmm_v2));
  // mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(cublasSetMatrix));
  // mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(cublasSscal));
  // mspHandlers->insert(CUBLAS_ROUTINE_HANDLER_PAIR(cublasDestroy));
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
