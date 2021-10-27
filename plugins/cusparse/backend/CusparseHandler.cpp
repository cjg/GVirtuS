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
 * Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>
 *             Department of Science and Technologies
 *
 */

#include <cstring>
#include <map>
#include <errno.h>
#include <cuda_runtime_api.h>
#include "CusparseHandler.h"

using namespace std;
using namespace log4cplus;

std::map<string, CusparseHandler::CusparseRoutineHandler> * CusparseHandler::mspHandlers = NULL;

extern "C" std::shared_ptr<CusparseHandler> create_t() {
    return std::make_shared<CusparseHandler>();
}


extern "C" int HandlerInit() {
    return 0;
}

CusparseHandler::CusparseHandler() {
    logger=Logger::getInstance(LOG4CPLUS_TEXT("CusparseHandler"));
    setLogLevel(&logger);
    Initialize();
}

CusparseHandler::~CusparseHandler() {

}

void CusparseHandler::setLogLevel(Logger *logger) {
        log4cplus::LogLevel logLevel=log4cplus::INFO_LOG_LEVEL;
        char * val = getenv("GVIRTUS_LOGLEVEL");
        std::string logLevelString =(val == NULL ? std::string("") : std::string(val));
        if(logLevelString != "") {
                logLevel=std::stoi(logLevelString);
        }
        logger->setLogLevel(logLevel);
}

bool CusparseHandler::CanExecute(std::string routine) {
    return mspHandlers->find(routine) != mspHandlers->end();

}

std::shared_ptr<Result> CusparseHandler::Execute(std::string routine, std::shared_ptr<Buffer> input_buffer) {
    LOG4CPLUS_DEBUG(logger,"Called " << routine);
    map<string, CusparseHandler::CusparseRoutineHandler>::iterator it;
    it = mspHandlers->find(routine);
    if (it == mspHandlers->end())
        throw "No handler for '" + routine + "' found!";
    try {
        return it->second(this, input_buffer);
    } catch (const char *ex) {
        LOG4CPLUS_DEBUG(logger,ex);
        LOG4CPLUS_DEBUG(logger,strerror(errno));
    }
    return NULL;
}

void CudnnHandler::Initialize(){
   if (mspHandlers != NULL)
        return;
    mspHandlers = new map<string, CudnnHandler::CusparseRoutineHandler> ();

    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetVersion));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetErrorString));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Create));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(Destroy));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(SetStream));
    mspHandlers->insert(CUSPARSE_ROUTINE_HANDLER_PAIR(GetStream));  
}

CUSPARSE_ROUTINE_HANDLER(GetVersion){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));

    size_t version = cusparseGetVersion();
    LOG4CPLUS_DEBUG(logger,"cusparseGetVersion Executed");
    return std::make_shared<Result>(version);
}

CUSPARSE_ROUTINE_HANDLER(GetErrorString){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetErrorString"));
    cusparseStatus_t cs = in->Get<cusparseStatus_t>();
    const char * s = cusparseGetErrorString(cs);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add((char *)s);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseGetErrorString Executed");
    return std::make_shared<Result>(CUSPARSE_STATUS_SUCCESS,out);
}

CUSPARSE_ROUTINE_HANDLER(Create){

    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Create"));
    cusparseHandle_t handle;
    cusparseStatus_t cs = cusparseCreate(&handle);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
         out->Add<cusparseHandle_t>(handle);
    } catch (string e){
                        LOG4CPLUS_DEBUG(logger,e);
                        return std::make_shared<Result>(CUSPARSE_STATUS_EXECUTION_FAILED);
    }
    LOG4CPLUS_DEBUG(logger,"cusparseCreate Executed");
    return std::make_shared<Result>(cs,out);

}

CUSPARSE_ROUTINE_HANDLER(Destroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Destroy"));

    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cusparseStatus_t cs = cusparseDestroy(handle);
    cout << "DEBUG - cusparseDestroy Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUSPARSE_ROUTINE_HANDLER(SetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cudaStream_t streamId = (cudaStream_t) in->Get<long long int>();

    cusparseStatus_t cs = cusparseSetStream(handle,streamId);
    cout << "DEBUG - cusparseSetStream Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUSPARSE_ROUTINE_HANDLER(GetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetStream"));
    cusparseHandle_t handle = (cusparseHandle_t)in->Get<long long int>();
    cudaStream_t *streamId;
    cudnnStatus_t cs = cusparseGetStream(handle,streamId);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
         out->Add<long long int>((long long int)*streamId);
    } catch (string e){
         LOG4CPLUS_DEBUG(logger,e);
         return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cusparseGetStream Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}
