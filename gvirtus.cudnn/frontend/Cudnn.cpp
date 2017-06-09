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

#include "CudnnFrontend.h"

using namespace std;

extern "C" size_t cudnnGetVersion(){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnGetVersion"); 
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t cudnnCreate(cudnnHandle_t *handle){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnCreate");
    if(CudnnFrontend::Success())
        *handle = CudnnFrontend::GetOutputVariable<cudnnHandle_t>();
    return CudnnFrontend::GetExitCode();
}

extern "C" const char * cudnnGetErrorString(cudnnStatus_t status){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<cudnnStatus_t>(status);
    CudnnFrontend::Execute("cudnnGetErrorString");
    return (const char *) CudnnFrontend::GetOutputHostPointer<char *>();
}

extern "C" cudnnStatus_t cudnnDestroy(cudnnHandle_t handle){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::Execute("cudnnDestroy");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)streamId);
    CudnnFrontend::Execute("cudnnSetStream");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::Execute("cudnnGetStream");
    if(CudnnFrontend::Success())
        *streamId = (cudaStream_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnCreateTensorDescriptor");
    if (CudnnFrontend::Success()){
        *tensorDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t cudnnSetTensor4dDescriptor( cudnnTensorDescriptor_t   tensorDesc,
                            cudnnTensorFormat_t format,
                            cudnnDataType_t dataType,
                            int n,
                            int c, int h, int w ) {
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(n);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);
    
    CudnnFrontend::Execute("cudnnSetTensor4dDescriptor");
    
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t cudnnSetTensor4dDescriptorEx( cudnnTensorDescriptor_t tensorDesc,
                              cudnnDataType_t dataType,
                              int n,
                              int c,
                              int h,
                              int w,
                              int nStride,
                              int cStride,
                              int hStride,
                              int wStride ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(n);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);
    
    CudnnFrontend::AddVariableForArguments<int>(nStride);
    CudnnFrontend::AddVariableForArguments<int>(cStride);
    CudnnFrontend::AddVariableForArguments<int>(hStride);
    CudnnFrontend::AddVariableForArguments<int>(wStride);
    
    CudnnFrontend::Execute("SetTensor4dDescriptorEx");
    return CudnnFrontend::GetExitCode();
}

extern "C"  cudnnStatus_t cudnnGetTensor4dDescriptor( cudnnTensorDescriptor_t tensorDesc,
                            cudnnDataType_t *dataType,
                            int *n,
                            int *c,
                            int *h,
                            int *w,
                            int *nStride,
                            int *cStride,
                            int *hStride,
                            int *wStride ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::Execute("cudnnGetTensor4dDescriptor");
    
    if(CudnnFrontend::Success()){
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *n = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
        *nStride = CudnnFrontend::GetOutputVariable<int>();
        *cStride = CudnnFrontend::GetOutputVariable<int>();
        *hStride = CudnnFrontend::GetOutputVariable<int>();
        *wStride = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}