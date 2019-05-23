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

#include "CublasHandler.h"
#include <iostream>
#include <cstdio>
#include <string>

using namespace std;
using namespace log4cplus;

CUBLAS_ROUTINE_HANDLER(Create) {
    cublasHandle_t handle=in->Get<cublasHandle_t>();
    cublasStatus_t cublas_status=cublasCreate(&handle);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    cout << "Handler create: " << handle << endl;
    out->Add<cublasHandle_t>(handle);
    return std::make_shared<Result>(cublas_status, out);
}

CUBLAS_ROUTINE_HANDLER(GetVersion_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));
    
    cublasHandle_t handle=(cublasHandle_t)in->Get<long long int>();
    int version;
    cublasStatus_t cs = cublasGetVersion(handle,&version);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<int>(version);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cublasGetVersion Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Create_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Create_v2"));
    cublasHandle_t handle ;//= in->Assign<cublasHandle_t>();
    cublasStatus_t cs = cublasCreate_v2(&handle);

    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<cublasHandle_t>(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    if(cs != CUBLAS_STATUS_SUCCESS)
        cout<<"----Error---"<<endl;
    cout << "DEBUG - cublasCreate_v2 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUBLAS_ROUTINE_HANDLER(Destroy_v2) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("Destroy_v2"));
    cublasHandle_t handle=(cublasHandle_t)in->Get<long long int>();
    cublasStatus_t cs=cublasDestroy(handle);
    cout << "DEBUG - cublasDestroy_v2 Executed"<<endl;
    //Buffer *out=new Buffer();
    return std::make_shared<Result>(cs);
}


CUBLAS_ROUTINE_HANDLER(SetVector){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetVector"));
    
    int n = in->Get<int>();
    int elemSize = in->Get<int>();
    int incx = in->Get<int>();
    int incy = in->Get<int>();
    
    void * y = in->GetFromMarshal<void*>();
    const void * x = in->AssignAll<char>();
    
    cublasStatus_t cs = cublasSetVector(n,elemSize,x,incx,y,incy);
    if(cs == CUBLAS_STATUS_NOT_INITIALIZED)
        cout<<"1"<<endl;
    if(cs == CUBLAS_STATUS_INVALID_VALUE)
        cout<<"2"<<endl;
    if(cs == CUBLAS_STATUS_MAPPING_ERROR)
        cout<<"3"<<endl;
    cout << "DEBUG - cublasSetVector Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(SetMatrix){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetMatrix"));
    
    int rows = (int)in->Get<int>();
    int cols = (int)in->Get<int>();
    int elemSize = (int)in->Get<int>();
    void * B = in->GetFromMarshal<void*>();
    int ldb = (int)in->Get<int>();
    int lda = (int)in->Get<int>();
    
    void * A = in->AssignAll<char>();
    cublasStatus_t cs = cublasSetMatrix(rows,cols,elemSize,A,lda,B,ldb);
    if(cs == CUBLAS_STATUS_NOT_INITIALIZED)
        cout<<"1"<<endl;
    if(cs == CUBLAS_STATUS_INVALID_VALUE)
        cout<<"2"<<endl;
    if(cs == CUBLAS_STATUS_MAPPING_ERROR)
        cout<<"3"<<endl;
    cout << "DEBUG - cublasSetVector Executed"<<endl;
    return std::make_shared<Result>(cs);
}


CUBLAS_ROUTINE_HANDLER(GetVector){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetVector"));
    
    int n = (int)in->Get<int>();
    int elemSize = (int)in->Get<int>();
    int incx = (int)in->Get<int>();
    int incy = (int)in->Get<int>();
    
    void * x = in->GetFromMarshal<void*>();
    void * y = in->Assign<void>();
    
    cublasStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        cs = cublasGetVector(n,elemSize,x,incx,y,incy);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    
    out->Add<char>((char*)y,n*elemSize);
    cout << "DEBUG - cublasGetVector Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUBLAS_ROUTINE_HANDLER(GetMatrix) {
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetMatrix"));
    
    int rows = (int)in->Get<int>();
    int cols = (int)in->Get<int>();
    int elemSize = (int)in->Get<int>();
    void * A = in->GetFromMarshal<void*>();
    int lda = (int)in->Get<int>();
    void * B = in->Assign<void>();
    int ldb = (int)in->Get<int>();
    
    cublasStatus_t cs;
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        cs = cublasGetMatrix(rows,cols,elemSize,A,lda,B,ldb);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cudaErrorMemoryAllocation);
    }
    out->Add<char>((char*)B,rows*cols*elemSize);
    cout << "DEBUG - cublasGetMatrix Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}
CUBLAS_ROUTINE_HANDLER(SetStream_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    cudaStream_t streamId = (cudaStream_t) in->Get<long long int>();
    
    cublasStatus_t cs = cublasSetStream_v2(handle,streamId);
    cout << "DEBUG - cublasSetStream Executed"<<endl;
    return std::make_shared<Result>(cs);
}

CUBLAS_ROUTINE_HANDLER(GetStream_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    cudaStream_t *streamId;
    cublasStatus_t cs = cublasGetStream_v2(handle,streamId);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try {
        out->Add<long long int>((long long int)*streamId);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cublasGetStream Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}

CUBLAS_ROUTINE_HANDLER(GetPointerMode_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("GetPointerMode_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    cublasPointerMode_t mode;
    cublasStatus_t cs = cublasGetPointerMode_v2(handle,&mode);
       std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<cublasPointerMode_t>(mode);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cublasGetPointerMode_v2 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}


CUBLAS_ROUTINE_HANDLER(SetPointerMode_v2){
    Logger logger=Logger::getInstance(LOG4CPLUS_TEXT("SetPointerMode_v2"));
    
    cublasHandle_t handle = (cublasHandle_t)in->Get<long long int>();
    cublasPointerMode_t mode = in->Get<cublasPointerMode_t>();
    cublasStatus_t cs = cublasSetPointerMode_v2(handle,mode);
        std::shared_ptr<Buffer> out = std::make_shared<Buffer>();

    try{
        out->Add<cublasPointerMode_t>(mode);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    cout << "DEBUG - cublasSetPointerMode_v2 Executed"<<endl;
    return std::make_shared<Result>(cs,out);
}


/*CUBLAS_ROUTINE_HANDLER(SetMatrix) {
    int rows=in->Assign<int>();
    int cols=in->Assign<int>();
    int elemSize=in->Assign<int>();
    const void *A=in->Get<void *>();
    int lda=in->Assign<int>();
    void *B=in->Assign<void *>();
    int ldb=in->Assign<int>();
    cublasStatus_t cublas_status=cublasSetMatrix(rows,cols,elemSize,A,lda,B,ldb);
    Buffer *out=new Buffer();
    return std::make_shared<Result>(cublas_status, out);
}

CUBLAS_ROUTINE_HANDLER(GetMatrix) {
    /*int rows=in->Assign<int>();
    int cols=in->Assign<int>();
    int elemSize=in->Assign<int>();
    const void *A=in->Get<void *>();
    int lda=in->Assign<int>();
    void *B=in->Assign<void *>();
    int ldb=in->Assign<int>();
    cublasStatus_t cublas_status=cublasGetMatrix(rows,cols,elemSize,A,lda,B,ldb);
    Buffer *out=new Buffer();
    return std::make_shared<Result>(cublas_status, out);   
}

CUBLAS_ROUTINE_HANDLER(Destroy) {
    cublasHandle_t handle=in->Get<cublasHandle_t>();
    cublasStatus_t cublas_status=cublasDestroy(handle);

    Buffer *out=new Buffer();
    return std::make_shared<Result>(cublas_status, out);
}*/ 
