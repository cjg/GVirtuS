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
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnHandle_t>(handle);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(CUDNN_STATUS_EXECUTION_FAILED);
    }
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetVersion){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetVersion"));
    
    size_t version = cudnnGetVersion();
    return std::make_shared<Result>(version);
}

CUDNN_ROUTINE_HANDLER(GetErrorString){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetErrorString"));
    cudnnStatus_t cs = in->Get<cudnnStatus_t>();
    const char * s = cudnnGetErrorString(cs);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add((char *)s);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(CUDNN_STATUS_EXECUTION_FAILED);
    }
    return std::make_shared<Result>(CUDNN_STATUS_SUCCESS,out);
}

CUDNN_ROUTINE_HANDLER(Destroy){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("Destroy"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroy(handle);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetStream){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetStream"));
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudaStream_t streamId = (cudaStream_t) in->Get<long long int>();
    
    cudnnStatus_t cs = cudnnSetStream(handle,streamId);
    return std::make_shared<Result>(cs);
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
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
}


CUDNN_ROUTINE_HANDLER(CreateTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateTensorDescriptor"));
    
    cudnnTensorDescriptor_t tensorDesc;
    cudnnStatus_t cs = cudnnCreateTensorDescriptor(&tensorDesc);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnTensorDescriptor_t>(tensorDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
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
    return std::make_shared<Result>(cs);
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
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(GetTensor4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetTensor4dDescriptor"));
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    
    cudnnDataType_t dataType;
    int n,c,h,w;
    int nStride,cStride,hStride,wStride;
    
    cudnnStatus_t cs = cudnnGetTensor4dDescriptor(tensorDesc,&dataType,&n,&c,&h,&w,&nStride,&cStride,&hStride,&wStride);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
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
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetTensorNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensorNdDescriptor"));
    
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    int nbDims = in->Get<int>();
    int *dimA = in->Assign<int>();
    int *strideA = in->Assign<int>();
    
    cudnnStatus_t cs = cudnnSetTensorNdDescriptor(tensorDesc,dataType,nbDims,dimA,strideA);
    return std::make_shared<Result>(cs);
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
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnDataType_t>(dataType);
        out->Add<int>(nbDims);
        out->Add<int>(dimA);
        out->Add<int>(strideA);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(DestroyTensorDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestroyTensorDescriptor"));
    
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    cudnnStatus_t cs = cudnnDestroyTensorDescriptor(tensorDesc);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(TransformTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("TransformTensor"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    void * alpha = in->Assign<void>();
    cudnnTensorDescriptor_t xDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * x = in->Assign<void>();
    void * beta = in->Assign<void>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();
    
    cudnnStatus_t cs = cudnnTransformTensor(handle,alpha,xDesc,x,beta,yDesc,y);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(AddTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("AddTensor"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    void * alpha = in->Assign<void>();
    cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * A = in->Assign<void>();
    void * beta = in->Assign<void>();
    cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * C = in->Assign<void>();
    
    cudnnStatus_t cs = cudnnAddTensor(handle,alpha,aDesc,A,beta,cDesc,C);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(OpTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("OpTensor"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnOpTensorDescriptor_t opTensorDesc = (cudnnOpTensorDescriptor_t)in->Get<long long int>();
    const void * alpha1 = in->Assign<void>();
    cudnnTensorDescriptor_t aDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * A = in->Assign<void>();
    void * alpha2 = in->Assign<void>();
    cudnnTensorDescriptor_t bDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * B = in->Assign<void>();
    void * beta = in->Assign<void>();
    cudnnTensorDescriptor_t cDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * C = in->Assign<void>();
    
    cudnnStatus_t cs = cudnnOpTensor(handle,opTensorDesc,alpha1,aDesc,A,alpha2,bDesc,B,beta,cDesc,C);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetTensor"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    cudnnTensorDescriptor_t yDesc = (cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();
    void * valuePtr = in->Assign<void>();
    
    cudnnStatus_t cs = cudnnSetTensor(handle,yDesc,y,valuePtr);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(ScaleTensor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ScaleTensor"));
    
    cudnnHandle_t handle = (cudnnHandle_t)in->Get<long long int>();
    const cudnnTensorDescriptor_t yDesc = (const cudnnTensorDescriptor_t)in->Get<long long int>();
    void * y = in->Assign<void>();
    void * alpha = in->Assign<void>();
    
    cudnnStatus_t cs = cudnnScaleTensor(handle,yDesc,y,alpha);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(CreateFilterDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateFilterDescriptor"));
    
    cudnnFilterDescriptor_t * filterDesc = in->Assign<cudnnFilterDescriptor_t>();
    cudnnStatus_t cs = cudnnCreateFilterDescriptor(filterDesc);
    return std::make_shared<Result>(cs);
}

CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    cudnnTensorFormat_t  format = (cudnnTensorFormat_t) in->Get<long long int>();
    
    int k = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();
    
    cudnnStatus_t cs = cudnnSetFilter4dDescriptor(filterDesc,dataType,format,k,c,h,w);
    return std::make_shared<Result>(cs);
}


CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType;
    cudnnTensorFormat_t  format;
    
    int k,c,h,w;
    
    cudnnStatus_t cs = cudnnGetFilter4dDescriptor(filterDesc,&dataType,&format,&k,&c,&h,&w);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    
    try{
        out->Add<long long int>((long long int)dataType);
        out->Add<long long int>((long long int)format);
        out->Add<int>(k);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v3"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    
    int k = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();
    
    cudnnStatus_t cs = cudnnSetFilter4dDescriptor_v3(filterDesc,dataType,k,c,h,w);
    return std::make_shared<Result>(cs);
}


CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v3"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType;
    
    int k,c,h,w;
    
    cudnnStatus_t cs = cudnnGetFilter4dDescriptor_v3(filterDesc,&dataType,&k,&c,&h,&w);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    
    try{
        out->Add<long long int>((long long int)dataType);
        out->Add<int>(k);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
}


CUDNN_ROUTINE_HANDLER(SetFilter4dDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v4"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    cudnnTensorFormat_t  format = (cudnnTensorFormat_t) in->Get<long long int>();
    
    int k = in->Get<int>();
    int c = in->Get<int>();
    int h = in->Get<int>();
    int w = in->Get<int>();
    
    cudnnStatus_t cs = cudnnSetFilter4dDescriptor_v4(filterDesc,dataType,format,k,c,h,w);
    return std::make_shared<Result>(cs);
}


CUDNN_ROUTINE_HANDLER(GetFilter4dDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilter4dDescriptor_v4"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType;
    cudnnTensorFormat_t  format;
    
    int k,c,h,w;
    
    cudnnStatus_t cs = cudnnGetFilter4dDescriptor_v4(filterDesc,&dataType,&format,&k,&c,&h,&w);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    
    try{
        out->Add<long long int>((long long int)dataType);
        out->Add<long long int>((long long int)format);
        out->Add<int>(k);
        out->Add<int>(c);
        out->Add<int>(h);
        out->Add<int>(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return std::make_shared<Result>(cs);
    }
    return std::make_shared<Result>(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    cudnnTensorFormat_t  format = (cudnnTensorFormat_t) in->Get<long long int>();
    
    int nbDims = in->Get<int>();
    int * filterDimA = in->Assign<int>();
    
    cudnnStatus_t cs = cudnnSetFilterNdDescriptor(filterDesc,dataType,format,nbDims,filterDimA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<long long int>((long long int)filterDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor"));
    
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();
    
    cudnnTensorFormat_t  format;
    
    cudnnStatus_t cs = cudnnGetFilterNdDescriptor(wDesc,nbDimsRequested,dataType,&format,nbDims,filterDimA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<long long int>(format);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor_v3"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    
    int nbDims = in->Get<int>();
    int * filterDimA = in->Assign<int>();
    
    cudnnStatus_t cs = cudnnSetFilterNdDescriptor_v3(filterDesc,dataType,nbDims,filterDimA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<long long int>((long long int)filterDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v3){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor"));
    
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();
    
    
    cudnnStatus_t cs = cudnnGetFilterNdDescriptor_v3(wDesc,nbDimsRequested,dataType,nbDims,filterDimA);
    return new Result(cs);
}

CUDNN_ROUTINE_HANDLER(SetFilterNdDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetFilterNdDescriptor_v4"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    cudnnDataType_t dataType = (cudnnDataType_t) in->Get<long long int>();
    cudnnTensorFormat_t  format = (cudnnTensorFormat_t) in->Get<long long int>();
    
    int nbDims = in->Get<int>();
    int * filterDimA = in->Assign<int>();
    
    cudnnStatus_t cs = cudnnSetFilterNdDescriptor_v4(filterDesc,dataType,format,nbDims,filterDimA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<long long int>((long long int)filterDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}


CUDNN_ROUTINE_HANDLER(GetFilterNdDescriptor_v4){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetFilterNdDescriptor_v4"));
    
    cudnnFilterDescriptor_t wDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    int nbDimsRequested = in->Get<int>();
    cudnnDataType_t *dataType = in->Assign<cudnnDataType_t>();
    int *nbDims = in->Assign<int>();
    int *filterDimA = in->Assign<int>();
    
    cudnnTensorFormat_t  format;
    
    cudnnStatus_t cs = cudnnGetFilterNdDescriptor_v4(wDesc,nbDimsRequested,dataType,&format,nbDims,filterDimA);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<long long int>(format);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(DestroyFilterDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("DestoryFilterDescriptor"));
    
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t)in->Get<long long int>();
    
    cudnnStatus_t cs = cudnnDestroyFilterDescriptor(filterDesc);
    return new Result(cs);
}

CUDNN_ROUTINE_HANDLER(CreateConvolutionDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("CreateConvolutionDescriptor"));

    cudnnConvolutionDescriptor_t convDesc;
    cudnnStatus_t cs = cudnnCreateConvolutionDescriptor(&convDesc);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetConvolution2dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolution2dDescriptor")); 
   
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int padh = in->Get<int>();
    int padw = in->Get<int>();
    int u = in->Get<int>();
    int v = in->Get<int>();
    int upscalex = in->Get<int>();
    int upscaley = in->Get<int>();
    cudnnConvolutionMode_t mode = in->BackGet<cudnnConvolutionMode_t>();
    
    cudnnStatus_t cs = cudnnSetConvolution2dDescriptor(convDesc,padh,padw,u,v,upscalex,upscaley,mode);
    
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetConvolution2dDescriptor){
   Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolution2dDescriptor")); 
   
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int padh,padw,u,v,upscalex,upscaley;
    cudnnConvolutionMode_t mode;
    
    cudnnStatus_t cs = cudnnGetConvolution2dDescriptor(convDesc,&padh,&padw,&u,&v,&upscalex,&upscaley,&mode);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try{
        out->Add(padh);
        out->Add(padw);
        out->Add(u);
        out->Add(v);
        out->Add(upscalex);
        out->Add(upscaley);
    } catch(string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(GetConvolution2dForwardOutputDim){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("GetConvolution2dForwardOutputDim"));
    
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    cudnnTensorDescriptor_t tensorDesc = (cudnnTensorDescriptor_t) in->Get<long long int>();
    cudnnFilterDescriptor_t filterDesc = (cudnnFilterDescriptor_t) in->Get<long long int>();
    
    int n,c,h,w;
    
    cudnnStatus_t cs = cudnnGetConvolution2dForwardOutputDim(convDesc,tensorDesc,filterDesc,&n,&c,&h,&w);
    std::shared_ptr<Buffer> out = std::make_shared<Buffer>();
    try {
        out->Add(n);
        out->Add(c);
        out->Add(h);
        out->Add(w);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}

CUDNN_ROUTINE_HANDLER(SetConvolutionNdDescriptor){
    Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("SetConvolutionNdDescriptor"));
    
    cudnnConvolutionDescriptor_t convDesc = (cudnnConvolutionDescriptor_t)in->Get<long long int>();
    int arrayLength = in->Get<int>();
    int * padA = in->Assign<int>();
    int * filterStrideA = in->Assign<int>();
    int * upscaleA = in->Assign<int>();
    cudnnConvolutionMode_t mode = in->Get<cudnnConvolutionMode_t>();
    cudnnDataType_t dataType = in->Get<cudnnDataType_t>();
    
    cudnnStatus_t cs = cudnnSetConvolutionNdDescriptor(convDesc,arrayLength,padA,filterStrideA,upscaleA,mode,dataType);
    
    try{
        out->Add<cudnnConvolutionDescriptor_t>(convDesc);
    } catch (string e){
        LOG4CPLUS_DEBUG(logger,e);
        return new Result(cs);
    }
    return new Result(cs,out);
}