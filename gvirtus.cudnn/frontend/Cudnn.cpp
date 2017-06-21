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

extern "C" size_t CUDNNWINAPI cudnnGetVersion(){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnGetVersion"); 
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnCreate");
    if(CudnnFrontend::Success())
        *handle = CudnnFrontend::GetOutputVariable<cudnnHandle_t>();
    return CudnnFrontend::GetExitCode();
}

extern "C" const char * CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<cudnnStatus_t>(status);
    CudnnFrontend::Execute("cudnnGetErrorString");
    return (const char *) CudnnFrontend::GetOutputHostPointer<char *>();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::Execute("cudnnDestroy");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)streamId);
    CudnnFrontend::Execute("cudnnSetStream");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::Execute("cudnnGetStream");
    if(CudnnFrontend::Success())
        *streamId = (cudaStream_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnCreateTensorDescriptor");
    if (CudnnFrontend::Success()){
        *tensorDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor( cudnnTensorDescriptor_t   tensorDesc,
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

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx( cudnnTensorDescriptor_t tensorDesc,
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

extern "C"  cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor( cudnnTensorDescriptor_t tensorDesc,
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

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor( cudnnTensorDescriptor_t tensorDesc,
                            cudnnDataType_t dataType,
                            int nbDims,
                            const int *dimA,
                            const int *strideA){
    
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)dimA);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)strideA);
    
    CudnnFrontend::Execute("cudnnSetTensorNdDescriptor");
    
    return CudnnFrontend::GetExitCode();  
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t tensorDesc,
                            int nbDimsRequested,
                            cudnnDataType_t *dataType,
                            int *nbDims,
                            int *dimA,
                            int *strideA){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::Execute("cudnnGetTensorNdDescriptor");
    if(CudnnFrontend::Success()){
        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *nbDims = CudnnFrontend::GetOutputVariable<int>();
        dimA = CudnnFrontend::GetOutputHostPointer<int>();
        strideA = CudnnFrontend::GetOutputHostPointer<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int) tensorDesc);
    CudnnFrontend::Execute("cudnnDestroyTensorDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformTensor( cudnnHandle_t                  handle,
                                                const void                    *alpha,
                                                const cudnnTensorDescriptor_t  xDesc,
                                                const void                    *x,
                                                const void                    *beta,
                                                const cudnnTensorDescriptor_t  yDesc,
                                                void                          *y ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    
    CudnnFrontend::Execute("cudnnTransformTensor");
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnAddTensor(
                                cudnnHandle_t                       handle,
                                const void                         *alpha,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C ){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)A);
    CudnnFrontend::AddHostPointerForArguments((void*)beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)C);
    
    CudnnFrontend::Execute("cudnnAddTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnOpTensor(
                                cudnnHandle_t                       handle,
                                const cudnnOpTensorDescriptor_t     opTensorDesc,
                                const void                         *alpha1,
                                const cudnnTensorDescriptor_t       aDesc,
                                const void                         *A,
                                const void                         *alpha2,
                                const cudnnTensorDescriptor_t       bDesc,
                                const void                         *B,
                                const void                         *beta,
                                const cudnnTensorDescriptor_t       cDesc,
                                void                               *C ){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)opTensorDesc);
    CudnnFrontend::AddHostPointerForArguments(alpha1);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)A);
    CudnnFrontend::AddHostPointerForArguments((void*)alpha2);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)bDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)B);
    CudnnFrontend::AddHostPointerForArguments((void*)beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    CudnnFrontend::AddHostPointerForArguments((void*)C);
    
    CudnnFrontend::Execute("cudnnOpTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor(
                            cudnnHandle_t                 handle,
                            const cudnnTensorDescriptor_t yDesc,
                            void                          *y,
                            const void                    *valuePtr ){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddHostPointerForArguments((void*)valuePtr);
    
    CudnnFrontend::Execute("cudnnSetTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnScaleTensor( cudnnHandle_t                 handle,
                                            const cudnnTensorDescriptor_t yDesc,
                                            void                          *y,
                                            const void                    *alpha){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddHostPointerForArguments((void*)alpha);
    
    CudnnFrontend::Execute("cudnnScaleTensor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddHostPointerForArguments<cudnnFilterDescriptor_t>(filterDesc);
    CudnnFrontend::Execute("cudnnCreateFilterDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor( cudnnFilterDescriptor_t filterDesc,
                                                    cudnnDataType_t dataType,
                                                    cudnnTensorFormat_t  format,
                                                    int k,
                                                    int c, int h, int w ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)format);
    
    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);
    
    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor");
    
    return CudnnFrontend::GetExitCode();
}

extern "C"  CUDNNWINAPI cudnnStatus_t cudnnGetFilter4dDescriptor( cudnnFilterDescriptor_t filterDesc,
                                                                                cudnnDataType_t *dataType,
                                                                                cudnnTensorFormat_t  *format,
                                                                                int *k,
                                                                                int *c,
                                                                                int *h,
                                                                                int *w ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    
    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor");
    
    if(CudnnFrontend::Success()){
        *dataType = (cudnnDataType_t) CudnnFrontend::GetOutputVariable<long long int>();
        *format = (cudnnTensorFormat_t) CudnnFrontend::GetOutputVariable<long long int>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor_v3( cudnnFilterDescriptor_t filterDesc,
                                                    cudnnDataType_t dataType,
                                                    int k,
                                                    int c, int h, int w ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    
    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);
    
    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor_v3");
    
    return CudnnFrontend::GetExitCode();
}

extern "C"  CUDNNWINAPI cudnnStatus_t cudnnGetFilter4dDescriptor_v3( cudnnFilterDescriptor_t filterDesc,
                                                                    cudnnDataType_t *dataType,
                                                                    int *k,
                                                                    int *c,
                                                                    int *h,
                                                                    int *w ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    
    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor_v3");
    
    if(CudnnFrontend::Success()){
        *dataType = (cudnnDataType_t) CudnnFrontend::GetOutputVariable<long long int>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor_v4( cudnnFilterDescriptor_t filterDesc,
                                                    cudnnDataType_t dataType,
                                                    cudnnTensorFormat_t  format,
                                                    int k,
                                                    int c, int h, int w ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)format);
    
    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);
    
    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor_v4");
    
    return CudnnFrontend::GetExitCode();
}

extern "C"  CUDNNWINAPI cudnnStatus_t cudnnGetFilter4dDescriptor_v4( cudnnFilterDescriptor_t filterDesc,
                                                                                cudnnDataType_t *dataType,
                                                                                cudnnTensorFormat_t  *format,
                                                                                int *k,
                                                                                int *c,
                                                                                int *h,
                                                                                int *w ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    
    CudnnFrontend::Execute("cudnnGetFilter4dDescriptor_v4");
    
    if(CudnnFrontend::Success()){
        *dataType = (cudnnDataType_t) CudnnFrontend::GetOutputVariable<long long int>();
        *format = (cudnnTensorFormat_t) CudnnFrontend::GetOutputVariable<long long int>();
        *k = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor( cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t  dataType,
                                                                cudnnTensorFormat_t  format,
                                                                int nbDims,
                                                                const int* filterDimA){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)format);
    
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterDimA);
    
    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor");
    if(CudnnFrontend::Success())
        filterDesc = (cudnnFilterDescriptor_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor( const cudnnFilterDescriptor_t wDesc,
                                                                int nbDimsRequested,
                                                                cudnnDataType_t *dataType,
                                                                cudnnTensorFormat_t  *format,
                                                                int *nbDims,
                                                                int *filterDimA ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);
    
    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);
    
    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor");
    if(CudnnFrontend::Success()){
        *format = (cudnnTensorFormat_t) CudnnFrontend::GetOutputVariable<long long int>();
    }
    
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor_v3( cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t  dataType,
                                                                int nbDims,
                                                                const int* filterDimA){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterDimA);
    
    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor_v3");
    if(CudnnFrontend::Success())
        filterDesc = (cudnnFilterDescriptor_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor_v3( const cudnnFilterDescriptor_t wDesc,
                                                                int nbDimsRequested,
                                                                cudnnDataType_t *dataType,
                                                                int *nbDims,
                                                                int *filterDimA ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);
    
    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);
    
    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor_v3");
    
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilterNdDescriptor_v4( cudnnFilterDescriptor_t filterDesc,
                                                                cudnnDataType_t  dataType,
                                                                cudnnTensorFormat_t  format,
                                                                int nbDims,
                                                                const int* filterDimA){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dataType);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)format);
    
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int *)filterDimA);
    
    CudnnFrontend::Execute("cudnnSetFilterNdDescriptor_v4");
    if(CudnnFrontend::Success())
        filterDesc = (cudnnFilterDescriptor_t) CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterNdDescriptor_v4( const cudnnFilterDescriptor_t wDesc,
                                                                int nbDimsRequested,
                                                                cudnnDataType_t *dataType,
                                                                cudnnTensorFormat_t  *format,
                                                                int *nbDims,
                                                                int *filterDimA ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments(dataType);
    
    CudnnFrontend::AddHostPointerForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>(filterDimA);
    
    CudnnFrontend::Execute("cudnnGetFilterNdDescriptor_v4");
    if(CudnnFrontend::Success()){
        *format = (cudnnTensorFormat_t) CudnnFrontend::GetOutputVariable<long long int>();
    }
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int) filterDesc);
    CudnnFrontend::Execute("cudnnDestroyFilterDescriptor");
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnCreateConvolutionDescriptor");
    if(CudnnFrontend::Success())
        *convDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
    return CudnnFrontend::GetExitCode();
}
 

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolution2dDescriptor( cudnnConvolutionDescriptor_t convDesc,
                                                                    int pad_h,
                                                                    int pad_w,
                                                                    int u,
                                                                    int v,
                                                                    int upscalex,
                                                                    int upscaley,
                                                                    cudnnConvolutionMode_t mode ){
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<int>(pad_h);
    CudnnFrontend::AddVariableForArguments<int>(pad_w);
    CudnnFrontend::AddVariableForArguments<int>(u);
    CudnnFrontend::AddVariableForArguments<int>(v);
    CudnnFrontend::AddVariableForArguments<int>(upscalex);
    CudnnFrontend::AddVariableForArguments<int>(upscaley);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionMode_t>(mode);
    
    CudnnFrontend::Execute("cudnnSetConvolution2dDescriptor");
    if(CudnnFrontend::Success())
        convDesc = (cudnnConvolutionDescriptor_t)CudnnFrontend::GetOutputVariable<long long int>();
    return CudnnFrontend::GetExitCode();
}


extern "C" CUDNNWINAPI cudnnStatus_t cudnnGetConvolution2dDescriptor( const cudnnConvolutionDescriptor_t convDesc,
                                                                        int* pad_h,
                                                                        int* pad_w,
                                                                        int* u,
                                                                        int* v,
                                                                        int* upscalex,
                                                                        int* upscaley,
                                                                        cudnnConvolutionMode_t *mode ){
    
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    
    CudnnFrontend::Execute("cudnnGetConvolution2dDescriptor");
    if(CudnnFrontend::Success()){
        *pad_h = CudnnFrontend::GetOutputVariable<int>();
        *pad_w = CudnnFrontend::GetOutputVariable<int>();
        *u = CudnnFrontend::GetOutputVariable<int>();
        *v = CudnnFrontend::GetOutputVariable<int>();
        *upscalex = CudnnFrontend::GetOutputVariable<int>();
        *upscaley = CudnnFrontend::GetOutputVariable<int>();
        *mode = CudnnFrontend::GetOutputVariable<cudnnConvolutionMode_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t cudnnGetConvolution2dForwardOutputDim( const cudnnConvolutionDescriptor_t convDesc,
                                                                const cudnnTensorDescriptor_t inputTensorDesc,
                                                                const cudnnFilterDescriptor_t filterDesc,
                                                                int *n,
                                                                int *c,
                                                                int *h,
                                                                int *w ){
    
    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)inputTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    
    CudnnFrontend::Execute("cudnnGetConvolution2dForwardOutputDim");
    if(CudnnFrontend::Success()){
        *n = CudnnFrontend::GetOutputVariable<int>();
        *c = CudnnFrontend::GetOutputVariable<int>();
        *h = CudnnFrontend::GetOutputVariable<int>();
        *w = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}