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

#include <iostream>
#include <cstdio>
#include <string>

#include "CudnnHandler.h"

using namespace std;
using namespace log4cplus;

CUDNN_ROUTINE_HANDLER(Create){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Create"));
    
    cudnnHandle_t handle;
    cudnnStatus_t cs = cudnnCreate(&handle);
    Buffer * out = new Buffer();
    try{
        out->Add<cudnnHandle_t>(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(CUDNN_STATUS_EXECUTION_FAILED);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetVersion){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));
    
    size_t version = cudnnGetVersion();
    return new Result(version);
}

CUDNN_ROUTINE_HANDLER(GetErrorString){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetErrorString"));
    cudnnStatus_t cs = in->Get<cudnnStatus_t>();
    const char * s = cudnnGetErrorString(cs);
    Buffer * out = new Buffer();
    try{
        out->Add((char *)s);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(CUDNN_STATUS_EXECUTION_FAILED);
    }
    return new Result(CUDNN_STATUS_SUCCESS,out);
}

CUDNN_ROUTINE_HANDLER(Destroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Destroy"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroy(handle);
    return new Result(cs);
}

CUDNN_ROUTINE_HANDLER(SetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudaStream_t streamId = (cudaStream_t) in->Get<long long int>();
    
    cudnnStatus_t cs = cudnnSetStream(handle,streamId);
    return new Result(cs);
}

CUDNN_ROUTINE_HANDLER(GetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetStream"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudaStream_t *streamId;
    
    cudnnStatus_t cs = cudnnGetStream(handle,streamId);
    Buffer *out = new Buffer();
    try {
        out->Add<long long int>((long long int)*streamId);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}


CUDNN_ROUTINE_HANDLER(CreateTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateTensorDescriptor"));
    
    cudnnTensorDescriptor_t tensorDesc;
    cudnnStatus_t cs = cudnnCreateTensorDescriptor(&tensorDesc);
    Buffer * out = new Buffer();
    try {
        out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor4dDescriptor"));
    
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnTensorFormat_t format = in->Get<cudnnTensorFormat_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    
    int n = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();
    
    cudnnStatus_t cs = cudnnSetTensor4dDescriptor(tensorDesc,format,dataType,n,c,h,w);
    return new Result(cs);
}

CUDNN_ROUTINE_HANDLER(SetTensor4dDescriptorEx){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor4dDescriptor"));
    
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    
    int n = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();
    
    int nStride = in->Get<int>();
    int cStride = in->Get<int>();
    int hStride = in->Get<int>();
    int wStride = in->Get<int>();
    
    cudnnStatus_t cs = cudnnSetTensor4dDescriptorEx(tensorDesc,dataType,n,c,h,w,nStride,cStride,hStride,wStride);
    return new Result(cs);
}

CUDNN_ROUTINE_HANDLER(GetTensor4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensor4dDescriptor"));
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    
    cudnnDataType_t dataType;
    int n,c,h,w;
    int nStride,cStride,hStride,wStride;
    
    cudnnStatus_t cs = cudnnGetTensor4dDescriptor(tensorDesc,&dataType,&n,&c,&h,&w,&nStride,&cStride,&hStride,&wStride);
    Buffer * out = new Buffer();
    try{
        out->Add<cudnnDataType_t>(dataType);
        out->Add<int>(n);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
        out->Add<int>(nStride);
        out->Add<int>(cStride);
        out->Add<int>(hStride);
        out->Add<int>(wStride);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorNdDescriptor"));
    
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();
    int *strideA = in->Assign<int>();
    
    cudnnStatus_t cs = cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA);
    return new Result(cs);
}

CUDNN_ROUTINE_HANDLER(GetTensorNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensorNdDescriptor"));
    
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t dataType;
    int *nbDims;
    int *dimA;
    int *strideA;
    
    cudnnStatus_t cs = cudnnGetTensorNdDescriptor(tensorDesc,nbDimsRequested,&dataType,nbDims,dimA,strideA);
    Buffer * out = new Buffer();
    try{
        out->Add<cudnnDataType_t>(dataType);
        out->Add<int>(nbDims);
        out->Add<int>(dimA);
        out->Add<int>(strideA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}