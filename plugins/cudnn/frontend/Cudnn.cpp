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

extern "C" const char * CUDNNWINAPI cudnnGetErrorString(cudnnStatus_t status){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnStatus_t>(status);
    CudnnFrontend::Execute("cudnnGetErrorString");
    return (const char *) CudnnFrontend::GetOutputHostPointer<char *>();
}
/*
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t *handle){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddHostPointerForArguments<cudnnHandle_t>(handle);
    CudnnFrontend::Execute("cudnnCreate");
    if(CudnnFrontend::Success())
        *handle = CudnnFrontend::GetOutputVariable<cudnnHandle_t>();
    return CudnnFrontend::GetExitCode();
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
*/
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

    //printf("[cudnnSetTensor4dDescriptor] N, C, H, W: %d %d %d %d\n", n, c, h, w);

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<int>(n);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetTensor4dDescriptor");
    if(CudnnFrontend::Success()){
        tensorDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
    }
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
    if(CudnnFrontend::Success()){
        tensorDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C"  cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor( const cudnnTensorDescriptor_t tensorDesc,
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
    if(CudnnFrontend::Success()){
         tensorDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t tensorDesc,
                                                                  cudnnTensorFormat_t format,
                                                                  cudnnDataType_t dataType,
                                                                  int nbDims,
                                                                  const int *dimA){

     CudnnFrontend::Prepare();
    
     CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);
     CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
     CudnnFrontend::AddVariableForArguments<int>(nbDims);
     CudnnFrontend::AddHostPointerForArguments<int>((int*)dimA);
   
      if(CudnnFrontend::Success()){
        tensorDesc = (cudnnTensorDescriptor_t)CudnnFrontend::GetOutputVariable<long long int>();
    }
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

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(const cudnnTensorDescriptor_t tensorDesc,
                                                               size_t *size){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)tensorDesc);

    CudnnFrontend::Execute("cudnnGetTensorSizeInBytes");
    if(CudnnFrontend::Success()){
       *size = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int) tensorDesc);
    CudnnFrontend::Execute("cudnnDestroyTensorDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnInitTransformDest(const cudnnTensorTransformDescriptor_t transformDesc,
                                                            const cudnnTensorDescriptor_t srcDesc,
                                                            cudnnTensorDescriptor_t destDesc,
                                                            size_t *destSizeInBytes){
    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)transformDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc);

    CudnnFrontend::Execute("cudnnInitTransformDest");
    
    if(CudnnFrontend::Success()){
       destDesc = (cudnnTensorDescriptor_t)CudnnFrontend::GetOutputVariable<long long int>();
       *destSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateTensorTransformDescriptor(cudnnTensorTransformDescriptor_t *transformDesc){
    CudnnFrontend::Prepare();
   
    CudnnFrontend::Execute("cudnnCreateTensorTransformDescriptor");
   
    if(CudnnFrontend::Success()){
       *transformDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();   
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                                                       const uint32_t nbDims,
                                                                       const cudnnTensorFormat_t destFormat,
                                                                       const int32_t *padBeforeA,
                                                                       const int32_t *padAfterA,
                                                                       const uint32_t *foldA,
                                                                       const cudnnFoldingDirection_t direction){

   CudnnFrontend::Prepare();
   
   CudnnFrontend::AddVariableForArguments<uint32_t>(nbDims);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)destFormat);
   CudnnFrontend::AddHostPointerForArguments<int32_t>((int32_t*)padBeforeA);
   CudnnFrontend::AddHostPointerForArguments<int32_t>((int32_t*)padAfterA);
   CudnnFrontend::AddHostPointerForArguments<uint32_t>((uint32_t*)foldA);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)direction);

   CudnnFrontend::Execute("cudnnSetTensorTransformDescriptor");
   if(CudnnFrontend::Success()){
      transformDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
  }
   return CudnnFrontend::GetExitCode(); 
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc,
                                                                       uint32_t nbDimsRequested,
                                                                       cudnnTensorFormat_t *destFormat,
                                                                       int32_t *padBeforeA,
                                                                       int32_t *padAfterA,
                                                                       uint32_t *foldA,
                                                                       cudnnFoldingDirection_t *direction){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)transformDesc);
    CudnnFrontend::AddVariableForArguments<uint32_t>(nbDimsRequested);

    CudnnFrontend::Execute("cudnnGetTensorTransformDescriptor");
    if(CudnnFrontend::Success()){
        *destFormat = CudnnFrontend::GetOutputVariable<cudnnTensorFormat_t>();
        *padBeforeA = CudnnFrontend::GetOutputVariable<int32_t>();
        *padAfterA  = CudnnFrontend::GetOutputVariable<int32_t>();
        *foldA      = CudnnFrontend::GetOutputVariable<uint32_t>();
        *direction  = CudnnFrontend::GetOutputVariable<cudnnFoldingDirection_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorTransformDescriptor(cudnnTensorTransformDescriptor_t transformDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int >((long long int)transformDesc);

    CudnnFrontend::Execute("cudnnDestroyTensorTransformDescriptor");
 
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

    CudnnFrontend::Execute("cudnnTransformTensor");
    if(CudnnFrontend::Success()){
       y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformTensorEx(cudnnHandle_t handle,
                                                            const cudnnTensorTransformDescriptor_t transDesc,
                                                            const void *alpha,
                                                            const cudnnTensorDescriptor_t srcDesc,
                                                            const void *srcData,
                                                            const void *beta,
                                                            const cudnnTensorDescriptor_t destDesc,
                                                            void *destData){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)transDesc);
     CudnnFrontend::AddHostPointerForArguments(alpha);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc);
     CudnnFrontend::AddHostPointerForArguments(srcData);
     CudnnFrontend::AddHostPointerForArguments(beta);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)destDesc);
     CudnnFrontend::AddHostPointerForArguments(destData);
    
     CudnnFrontend::Execute("cudnnTransformTensorEx");
   
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFoldedConvBackwardDataDescriptors(const cudnnHandle_t handle,
                                                                               const cudnnFilterDescriptor_t filterDesc,
                                                                               const cudnnTensorDescriptor_t diffDesc,
                                                                               const cudnnConvolutionDescriptor_t convDesc,
                                                                               const cudnnTensorDescriptor_t gradDesc,
                                                                               const cudnnTensorFormat_t transformFormat,
                                                                               cudnnFilterDescriptor_t foldedFilterDesc,
                                                                               cudnnTensorDescriptor_t paddedDiffDesc,
                                                                               cudnnConvolutionDescriptor_t foldedConvDesc,
                                                                               cudnnTensorDescriptor_t foldedGradDesc,
                                                                               cudnnTensorTransformDescriptor_t filterFoldTransDesc,
                                                                               cudnnTensorTransformDescriptor_t diffPadTransDesc,
                                                                               cudnnTensorTransformDescriptor_t gradFoldTransDesc,
                                                                               cudnnTensorTransformDescriptor_t gradUnfoldTransDesc){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     
     CudnnFrontend::Execute("cudnnGetFoldedConvBackwardDataDescriptors");

     if(CudnnFrontend::Success()){
         foldedFilterDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
         paddedDiffDesc   = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
         foldedConvDesc   = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
         foldedGradDesc   = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
         filterFoldTransDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
         diffPadTransDesc    = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
         gradFoldTransDesc   = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>();
         gradUnfoldTransDesc = CudnnFrontend::GetOutputVariable<cudnnTensorTransformDescriptor_t>(); 
      }
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
    //CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments(*(float *)alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    //CudnnFrontend::AddVariableForArguments(A);
    CudnnFrontend::AddDevicePointerForArguments(A);
    CudnnFrontend::AddVariableForArguments(*(float *)beta);
    //CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    //CudnnFrontend::AddVariableForArguments(C);
    CudnnFrontend::AddDevicePointerForArguments(C);

    CudnnFrontend::Execute("cudnnAddTensor");
    if(CudnnFrontend::Success()){
        //TEMPORANEAMENTE NON FUNZIONANTE
        C = CudnnFrontend::GetOutputDevicePointer();
     }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t *opTensorDesc){

    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateOpTensorDescriptor");
    if(CudnnFrontend::Success()){
        *opTensorDesc = CudnnFrontend::GetOutputVariable<cudnnOpTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}
     
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                                                                cudnnOpTensorOp_t opTensorOp,
                                                                cudnnDataType_t opTensorCompType,
                                                                cudnnNanPropagation_t opTensorNanOpt){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnOpTensorOp_t>(opTensorOp);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(opTensorCompType);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(opTensorNanOpt);

    CudnnFrontend::Execute("cudnnSetOpTensorDescriptor");
    if(CudnnFrontend::Success()){
        opTensorDesc = CudnnFrontend::GetOutputVariable<cudnnOpTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetOpTensorDescriptor(const cudnnOpTensorDescriptor_t opTensorDesc,
                                                                cudnnOpTensorOp_t *opTensorOp,
                                                                cudnnDataType_t *opTensorCompType,
                                                                cudnnNanPropagation_t *opTensorNanOpt){


   CudnnFrontend::Prepare();
  
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)opTensorDesc);

   CudnnFrontend::Execute("cudnnGetOpTensorDescriptor");
   if(CudnnFrontend::Success()){
      *opTensorOp = CudnnFrontend::GetOutputVariable<cudnnOpTensorOp_t>();
      *opTensorCompType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
      *opTensorNanOpt   = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)opTensorDesc);

    CudnnFrontend::Execute("cudnnDestroyOpTensorDescriptor");

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

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t *reduceTensorDesc){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::Execute("cudnnCreateReduceTensorDescriptor");
    if(CudnnFrontend::Success()){
        *reduceTensorDesc = CudnnFrontend::GetOutputVariable<cudnnReduceTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                    cudnnReduceTensorOp_t reduceTensorOp,
                                                                    cudnnDataType_t reduceTensorCompType,
                                                                    cudnnNanPropagation_t reduceTensorNanOpt,
                                                                    cudnnReduceTensorIndices_t reduceTensorIndices,
                                                                    cudnnIndicesType_t reduceTensorIndicesType){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<cudnnReduceTensorOp_t>(reduceTensorOp);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(reduceTensorCompType);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(reduceTensorNanOpt);
    CudnnFrontend::AddVariableForArguments<cudnnReduceTensorIndices_t>(reduceTensorIndices);
    CudnnFrontend::AddVariableForArguments<cudnnIndicesType_t>(reduceTensorIndicesType);

    CudnnFrontend::Execute("cudnnSetReduceTensorDescriptor");
    if(CudnnFrontend::Success()){
       reduceTensorDesc = CudnnFrontend::GetOutputVariable<cudnnReduceTensorDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReduceTensorDescriptor(const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                    cudnnReduceTensorOp_t *reduceTensorOp,
                                                                    cudnnDataType_t *reduceTensorCompType,
                                                                    cudnnNanPropagation_t *reduceTensorNanOpt,
                                                                    cudnnReduceTensorIndices_t *reduceTensorIndices,
                                                                    cudnnIndicesType_t *reduceTensorIndicesType){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorNanOpt);
    
    CudnnFrontend::Execute("cudnnGetReduceTensorDescriptor");
    if(CudnnFrontend::Success()){
        *reduceTensorOp = CudnnFrontend::GetOutputVariable<cudnnReduceTensorOp_t>();
        *reduceTensorCompType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *reduceTensorIndices  = CudnnFrontend::GetOutputVariable<cudnnReduceTensorIndices_t>();
        *reduceTensorIndicesType = CudnnFrontend::GetOutputVariable<cudnnIndicesType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc){

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);

    CudnnFrontend::Execute("cudnnDestroyReduceTensorDescriptor");  

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionIndicesSize(cudnnHandle_t handle,
                                                                  const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                  const cudnnTensorDescriptor_t aDesc,
                                                                  const cudnnTensorDescriptor_t cDesc,
                                                                  size_t *sizeInBytes){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    
   CudnnFrontend::Execute("cudnnGetReductionIndicesSize");
   if(CudnnFrontend::Success()){
       *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetReductionWorkspaceSize(cudnnHandle_t handle,
                                                                    const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                                    const cudnnTensorDescriptor_t aDesc,
                                                                    const cudnnTensorDescriptor_t cDesc,
                                                                    size_t *sizeInBytes){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    
    CudnnFrontend::Execute("cudnnGetReductionWorkspaceSize");
    if(CudnnFrontend::Success()){
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnReduceTensor(cudnnHandle_t handle,
                                                       const cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                                       void *indices,
                                                       size_t indicesSizeInBytes,
                                                       void *workspace,
                                                       size_t workspaceSizeInBytes,
                                                       const void *alpha,
                                                       const cudnnTensorDescriptor_t aDesc,
                                                       const void *A,
                                                       const void *beta,
                                                       const cudnnTensorDescriptor_t cDesc,
                                                       void *C){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)reduceTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)indicesSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workspaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)aDesc);
    CudnnFrontend::AddHostPointerForArguments(A);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
    CudnnFrontend::AddHostPointerForArguments(C); 

    CudnnFrontend::Execute("cudnnReduceTensor");
    if(CudnnFrontend::Success()){
       indices = CudnnFrontend::GetOutputDevicePointer();
       C       = CudnnFrontend::GetOutputDevicePointer();
    }  
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

    CudnnFrontend::Execute("cudnnCreateFilterDescriptor");
    if(CudnnFrontend::Success()){
       *filterDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
     }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFilter4dDescriptor( cudnnFilterDescriptor_t filterDesc,
                                                    cudnnDataType_t dataType,
                                                    cudnnTensorFormat_t  format,
                                                    int k,
                                                    int c, int h, int w ){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<cudnnTensorFormat_t>(format);

    CudnnFrontend::AddVariableForArguments<int>(k);
    CudnnFrontend::AddVariableForArguments<int>(c);
    CudnnFrontend::AddVariableForArguments<int>(h);
    CudnnFrontend::AddVariableForArguments<int>(w);

    CudnnFrontend::Execute("cudnnSetFilter4dDescriptor");
    if(CudnnFrontend::Success()){
        filterDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
    }
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

#if CUDNN_VERSION < 6000
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
#endif

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

#if CUDNN_VERSION < 6000
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
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFilterSizeInBytes(const cudnnFilterDescriptor_t filterDesc, size_t *size){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);

    CudnnFrontend::Execute("cudnnGetFilterSizeInBytes");
    if(CudnnFrontend::Success()){
       *size = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int) filterDesc);
    CudnnFrontend::Execute("cudnnDestroyFilterDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformFilter(cudnnHandle_t handle,
                                                          const cudnnTensorTransformDescriptor_t transDesc,
                                                          const void *alpha,
                                                          const cudnnFilterDescriptor_t srcDesc,
                                                          const void *srcData,
                                                          const void *beta,
                                                          const cudnnFilterDescriptor_t destDesc,
                                                          void *destData){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)transDesc);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc);
    CudnnFrontend::AddHostPointerForArguments(srcData);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)destDesc);
    CudnnFrontend::AddHostPointerForArguments(destData);

    CudnnFrontend::Execute("cudnnTransformFilter");

    return CudnnFrontend::GetExitCode();
}

#if 0
extern "C" cudnnStatus_t CUDNNWINAPI cudnnReorderFilterAndBias(cudnnHandle_t handle,
                                                               const cudnnFilterDescriptor_t filterDesc,
                                                               cudnnReorderType_t reorderType,
                                                               const void *filterData,
                                                               void *reorderedFilterData,
                                                               int reorderBias,
                                                               const void *biasData,
                                                               void *reorderedBiasData){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
   CudnnFrontend::AddVariableForArguments<cudnnReorderType_t>(reorderType);
   CudnnFrontend::AddHostPointerForArguments(filterData);
   CudnnFrontend::AddHostPointerForArguments(reorderedFilterData);
   CudnnFrontend::AddVariableForArguments<int>(reorderBias);
   CudnnFrontend::AddHostPointerForArguments(biasData);
   CudnnFrontend::AddHostPointerForArguments(reorderedBiasData);

   CudnnFrontend::Execute("cudnnReorderFilterAndBias");
   
   return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc){
    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateConvolutionDescriptor");
    if(CudnnFrontend::Success()){
         *convDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
    }       
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mathType);

    CudnnFrontend::Execute("cudnnSetConvolutionMathType");

    return CudnnFrontend::GetExitCode(); 
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t *mathType){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

    CudnnFrontend::Execute("cudnnGetConvolutionMathType");
    if(CudnnFrontend::Success()){
       *mathType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount){

     CudnnFrontend::Prepare();
     
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<int>(groupCount);
     
     CudnnFrontend::Execute("cudnnSetConvolutionGroupCount");
     
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int *groupCount){

     CudnnFrontend::Prepare();
    
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

     CudnnFrontend::Execute("cudnnGetConvolutionGroupCount");
     if(CudnnFrontend::Success()){
         *groupCount = CudnnFrontend::GetOutputVariable<int>();
     }
     return CudnnFrontend::GetExitCode();
}

#if 0
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                                    cudnnReorderType_t reorderType){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnReorderType_t>(reorderType);

    CudnnFrontend::Execute("cudnnSetConvolutionReorderType");
 
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionReorderType(cudnnConvolutionDescriptor_t convDesc,
                                                                    cudnnReorderType_t *reorderType){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

    CudnnFrontend::Execute("cudnnGetConvolutionReorderType");
    if(CudnnFrontend::Success()){
        *reorderType = CudnnFrontend::GetOutputVariable<cudnnReorderType_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI  cudnnSetConvolution2dDescriptor( cudnnConvolutionDescriptor_t convDesc,
                                                                    int pad_h,
                                                                    int pad_w,
                                                                    int u,
                                                                    int v,
                                                                    int upscalex,
                                                                    int upscaley,
                                                                    cudnnConvolutionMode_t mode,
                                                                    cudnnDataType_t computeType){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<int>(pad_h);
    CudnnFrontend::AddVariableForArguments<int>(pad_w);
    CudnnFrontend::AddVariableForArguments<int>(u);
    CudnnFrontend::AddVariableForArguments<int>(v);
    CudnnFrontend::AddVariableForArguments<int>(upscalex);
    CudnnFrontend::AddVariableForArguments<int>(upscaley);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(computeType);

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
                                                                        cudnnConvolutionMode_t *mode,
                                                                        cudnnDataType_t* computeType ){

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
        *computeType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
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

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetConvolutionNdDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                                                                   int arrayLength,
                                                                                   const int *padA,
                                                                                   const int *filterStrideA,
                                                                                   const int *dilationA,
                                                                                   cudnnConvolutionMode_t mode,
                                                                                   cudnnDataType_t computeType){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<int>(arrayLength);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)padA);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)filterStrideA);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)dilationA);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(computeType);

    CudnnFrontend::Execute("cudnnSetConvolutionNdDescriptor");
    if(CudnnFrontend::Success()){
        convDesc = CudnnFrontend::GetOutputVariable<cudnnConvolutionDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdDescriptor(const cudnnConvolutionDescriptor_t convDesc,
                                                                     int arrayLengthRequested,
                                                                     int *arrayLength,
                                                                     int *padA,
                                                                     int *strideA,
                                                                     int *dilationA,
                                                                     cudnnConvolutionMode_t *mode,
                                                                     cudnnDataType_t *computeType){

    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<int>(arrayLengthRequested);
    
    CudnnFrontend::Execute("cudnnGetConvolutionNdDescriptor");
    if(CudnnFrontend::Success()){
        arrayLengthRequested = CudnnFrontend::GetOutputVariable<int>();
        *arrayLength = CudnnFrontend::GetOutputVariable<int>();
        *strideA     = CudnnFrontend::GetOutputVariable<int>();
        *dilationA   = CudnnFrontend::GetOutputVariable<int>();
        *mode        = CudnnFrontend::GetOutputVariable<cudnnConvolutionMode_t>();
        *computeType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionNdForwardOutputDim(const cudnnConvolutionDescriptor_t convDesc,
                                                                           const cudnnTensorDescriptor_t inputTensorDesc,
                                                                           const cudnnFilterDescriptor_t filterDesc,
                                                                           int nbDims,
                                                                           int *tensorOuputDimA){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)inputTensorDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);

    CudnnFrontend::Execute("cudnnGetConvolutionNdForwardOutputDim");
    if(CudnnFrontend::Success()){
       *tensorOuputDimA =  CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

    CudnnFrontend::Execute("cudnnDestroyConvolutionDescriptor");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle_t handle, int *count){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int >((long long int)handle);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithmMaxCount");
    if(CudnnFrontend::Success()){
        *count = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                                                          const cudnnTensorDescriptor_t xDesc,
                                                                          const cudnnFilterDescriptor_t wDesc,
                                                                          const cudnnConvolutionDescriptor_t convDesc,
                                                                          const cudnnTensorDescriptor_t yDesc,
                                                                          const int requestedAlgoCount,
                                                                          int *returnedAlgoCount,
                                                                          cudnnConvolutionFwdAlgoPerf_t *perfResults){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

    CudnnFrontend::Execute("cudnnFindConvolutionForwardAlgorithm");
    if(CudnnFrontend::Success()){
         *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
         *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionForwardAlgorithmEx(cudnnHandle_t handle,
                                                                            const cudnnTensorDescriptor_t xDesc,
                                                                            const void *x,
                                                                            const cudnnFilterDescriptor_t wDesc,
                                                                            const void *w,
                                                                            const cudnnConvolutionDescriptor_t convDesc,
                                                                            const cudnnTensorDescriptor_t yDesc,
                                                                            void *y,
                                                                            const int requestedAlgoCount,
                                                                            int *returnedAlgoCount,
                                                                            cudnnConvolutionFwdAlgoPerf_t *perfResults,
                                                                            void *workSpace,
                                                                            size_t workSpaceSizeInBytes){

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindConvolutionForwardAlgorithmEx");
    if(CudnnFrontend::Success()){
        y = CudnnFrontend::GetOutputDevicePointer();
       *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
       *perfResults = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                                                              const cudnnTensorDescriptor_t xDesc,
                                                                              const cudnnFilterDescriptor_t wDesc,
                                                                              const cudnnConvolutionDescriptor_t convDesc,
                                                                              const cudnnTensorDescriptor_t yDesc,
                                                                              cudnnConvolutionFwdPreference_t preference,
                                                                              size_t memoryLimitInBytes,
                                                                              cudnnConvolutionFwdAlgo_t *algo){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdPreference_t>(preference);
    CudnnFrontend::AddVariableForArguments<size_t>(memoryLimitInBytes);
    
    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithm");
    if(CudnnFrontend::Success()){
        *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgo_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 7000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                                                            const cudnnTensorDescriptor_t srcDesc,
                                                                            const cudnnFilterDescriptor_t filterDesc,
             						                    const cudnnConvolutionDescriptor_t convDesc,
								            const cudnnTensorDescriptor_t destDesc,
									    const int requestedAlgoCount,
 									    int *returnedAlgoCount,
								     	    cudnnConvolutionFwdAlgoPerf_t *perfResults){


    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc); 
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)destDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardAlgorithm_v7");
    if(CudnnFrontend::Success()){
        *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
        *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionFwdAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                                                                  const cudnnTensorDescriptor_t xDesc,
                                                                                  const cudnnFilterDescriptor_t wDesc,
                                                                                  const cudnnConvolutionDescriptor_t convDesc,
                                                                                  const cudnnTensorDescriptor_t yDesc,
                                                                                  cudnnConvolutionFwdAlgo_t algo,
                                                                                   size_t *sizeInBytes){
    
    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);

    CudnnFrontend::Execute("cudnnGetConvolutionForwardWorkspaceSize");
    if(CudnnFrontend::Success()){
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionForward(cudnnHandle_t handle,
                                                             const void *alpha,
                                                             const cudnnTensorDescriptor_t xDesc,
                                                             const void *x,
                                                             const cudnnFilterDescriptor_t wDesc,
                                                             const void *w,
                                                             const cudnnConvolutionDescriptor_t convDesc,
                                                             cudnnConvolutionFwdAlgo_t algo,
                                                             void *workSpace,
                                                             size_t workSpaceSizeInBytes,
                                                             const void *beta,
                                                             const cudnnTensorDescriptor_t yDesc,
                                                             void *y){


    //printf("ALPHA IN FRONTEND INIZIALE: %f\n", *(float  *)alpha);

    CudnnFrontend::Prepare();
  
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments(*(float *)alpha);
    //CudnnFrontend::AddHostPointerForArguments(&alpha);
    //CudnnFrontend::AddDevicePointerForArguments(&alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddVariableForArguments(*(float *)beta);
    //CudnnFrontend::AddHostPointerForArguments(&beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);

    //printf("ALPHA IN FRONTEND FINALE: %f\n", *(float *) alpha);

    CudnnFrontend::Execute("cudnnConvolutionForward");
    if(CudnnFrontend::Success()){
        y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBiasActivationForward(cudnnHandle_t handle,
                                                                           const void *alpha1,
                                                                           const cudnnTensorDescriptor_t xDesc,
                                                                           const void *x,
                                                                           const cudnnFilterDescriptor_t wDesc,
                                                                           const void *w,
                                                                           const cudnnConvolutionDescriptor_t convDesc,
                                                                           cudnnConvolutionFwdAlgo_t algo,
                                                                           void *workSpace,
                                                                           size_t workSpaceSizeInBytes,
                                                                           const void *alpha2,
                                                                           const cudnnTensorDescriptor_t zDesc,
                                                                           const void *z,
                                                                           const cudnnTensorDescriptor_t biasDesc,
                                                                           const void *bias,
                                                                           const cudnnActivationDescriptor_t activationDesc,
                                                                           const cudnnTensorDescriptor_t yDesc,
                                                                           void *y){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddHostPointerForArguments(alpha1);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddHostPointerForArguments(w);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionFwdAlgo_t>(algo);
     CudnnFrontend::AddHostPointerForArguments(workSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddHostPointerForArguments(alpha2);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)zDesc);
     CudnnFrontend::AddHostPointerForArguments(z);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)biasDesc);
     CudnnFrontend::AddHostPointerForArguments(bias);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddHostPointerForArguments(y);
     
     CudnnFrontend::Execute("cudnnConvolutionBiasActivationForward");
     if(CudnnFrontend::Success()){
         y = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                                                                  const void *alpha,
                                                                  const  cudnnTensorDescriptor_t dyDesc,
                                                                  const void *dy,
                                                                  const void *beta,
                                                                  const cudnnTensorDescriptor_t dbDesc,
                                                                  void *db){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments(*(float *)alpha);
     //CudnnFrontend::AddHostPointerForArguments(alpha);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddDevicePointerForArguments(dy);
     CudnnFrontend::AddVariableForArguments(*(float *)beta);
     //CudnnFrontend::AddHostPointerForArguments(beta);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dbDesc);
     CudnnFrontend::AddDevicePointerForArguments(db);
     
     CudnnFrontend::Execute("cudnnConvolutionBackwardBias");
     if(CudnnFrontend::Success()){
        db = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnnHandle_t handle, int *count){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithmMaxCount");
    if(CudnnFrontend::Success()){
       *count = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                                                                 const cudnnTensorDescriptor_t xDesc,
                                                                                 const cudnnTensorDescriptor_t dyDesc,
                                                                                 const cudnnConvolutionDescriptor_t convDesc,
                                                                                 const cudnnFilterDescriptor_t dwDesc,
                                                                                 const int requestedAlgoCount,
                                                                                 int *returnedAlgoCount,
                                                                                 cudnnConvolutionBwdFilterAlgoPerf_t *perfResults){

    CudnnFrontend::Prepare();
  
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

    CudnnFrontend::Execute("cudnnFindConvolutionBackwardFilterAlgorithm");
    if(CudnnFrontend::Success()){
       *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
       *perfResults = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnnHandle_t handle,
                                                                                   const cudnnTensorDescriptor_t xDesc,
                                                                                   const void *x,
                                                                                   const cudnnTensorDescriptor_t dyDesc,
                                                                                   const void *y,
                                                                                   const cudnnConvolutionDescriptor_t convDesc,
                                                                                   const cudnnFilterDescriptor_t dwDesc,
                                                                                   void *dw,
                                                                                   const int requestedAlgoCount,
                                                                                   int *returnedAlgoCount,
                                                                                   cudnnConvolutionBwdFilterAlgoPerf_t *perfResults,
                                                                                   void *workSpace,
                                                                                   size_t workSpaceSizeInBytes){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
    CudnnFrontend::AddHostPointerForArguments(dw);
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnFindConvolutionBackwardFilterAlgorithmEx");
    if(CudnnFrontend::Success()){
       dw = CudnnFrontend::GetOutputDevicePointer();
       *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
       *perfResults = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgoPerf_t>();
    }
    return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                                                                const cudnnTensorDescriptor_t xDesc,
                                                                                const cudnnTensorDescriptor_t dyDesc,
                                                                                const cudnnConvolutionDescriptor_t convDesc,
                                                                                const cudnnFilterDescriptor_t dwDesc,
                                                                                cudnnConvolutionBwdFilterPreference_t preference,
                                                                                size_t memoryLimitInBytes,
                                                                                cudnnConvolutionBwdFilterAlgo_t *algo){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterPreference_t>(preference);
     CudnnFrontend::AddVariableForArguments<size_t>(memoryLimitInBytes);

     CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithm");
     if(CudnnFrontend::Success()){
         *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgo_t>();
     }
     return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 7000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_t handle,
                                                                                   const cudnnTensorDescriptor_t srcDesc,
                                                                                   const cudnnTensorDescriptor_t diffDesc,
                                                                                   const cudnnConvolutionDescriptor_t convDesc,
                                                                                   const cudnnFilterDescriptor_t gradDesc,
                                                                                   const int requestedAlgoCount,
                                                                                   int *returnedAlgoCount,
                                                                                   cudnnConvolutionBwdFilterAlgoPerf_t *perfResults){

      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)srcDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)diffDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)gradDesc);
      CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
      
      CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterAlgorithm_v7");
      if(CudnnFrontend::Success()){
          *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
          *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdFilterAlgoPerf_t>();
      }
      return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
                                                                                    const cudnnTensorDescriptor_t xDesc,
                                                                                    const cudnnTensorDescriptor_t dyDesc,
                                                                                    const cudnnConvolutionDescriptor_t convDesc,
                                                                                    const cudnnFilterDescriptor_t gradDesc,
                                                                                    cudnnConvolutionBwdFilterAlgo_t algo,
                                                                                    size_t *sizeInBytes){


     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)gradDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterAlgo_t>(algo);
     

    CudnnFrontend::Execute("cudnnGetConvolutionBackwardFilterWorkspaceSize");
    if(CudnnFrontend::Success()){
      *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
                                                                    const void *alpha,
                                                                    const cudnnTensorDescriptor_t xDesc,
                                                                    const void *x,
                                                                    const cudnnTensorDescriptor_t dyDesc,
                                                                    const void *dy,
                                                                    const cudnnConvolutionDescriptor_t convDesc,
                                                                    cudnnConvolutionBwdFilterAlgo_t algo,
                                                                    void *workSpace,
                                                                    size_t workSpaceSizeInBytes,
                                                                    const void *beta,
                                                                    const cudnnFilterDescriptor_t dwDesc,
                                                                    void *dw){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments(*(float *)alpha);
     //CudnnFrontend::AddHostPointerForArguments(alpha);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddDevicePointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddDevicePointerForArguments(dy);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdFilterAlgo_t>(algo);
     CudnnFrontend::AddDevicePointerForArguments(workSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddVariableForArguments(*(float *)beta);
     //CudnnFrontend::AddHostPointerForArguments(beta);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
     CudnnFrontend::AddDevicePointerForArguments(dw);

     CudnnFrontend::Execute("cudnnConvolutionBackwardFilter");
     if(CudnnFrontend::Success()){
        dw = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, int *count){
     
     CudnnFrontend::Prepare();
     
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);

     CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithmMaxCount");
     if(CudnnFrontend::Success()){
         *count = CudnnFrontend::GetOutputVariable<int>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                                                               const cudnnFilterDescriptor_t wDesc,
                                                                               const cudnnTensorDescriptor_t dyDesc,
                                                                               const cudnnConvolutionDescriptor_t convDesc,
                                                                               const cudnnTensorDescriptor_t dxDesc,
                                                                               const int requestedAlgoCount,
                                                                               int *returnedAlgoCount,
                                                                               cudnnConvolutionBwdDataAlgoPerf_t *perfResults){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
     CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

     CudnnFrontend::Execute("cudnnFindConvolutionBackwardDataAlgorithm");
     if(CudnnFrontend::Success()){
         *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
         *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgoPerf_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindConvolutionBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                						 const cudnnFilterDescriptor_t wDesc,
										 const void *w,
 										 const cudnnTensorDescriptor_t dyDesc,
										 const void *dy,
										 const cudnnConvolutionDescriptor_t convDesc,
										 const cudnnTensorDescriptor_t dxDesc,
     										 void *dx,
										 const int requestedAlgoCount,
										 int *returnedAlgoCount,
										 cudnnConvolutionBwdDataAlgoPerf_t *perfResults,
										 void *workSpace,
										 size_t workSpaceSizeInBytes){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
   CudnnFrontend::AddHostPointerForArguments(w);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
   CudnnFrontend::AddHostPointerForArguments(dy);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
   CudnnFrontend::AddHostPointerForArguments(dx);
   CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
   CudnnFrontend::AddHostPointerForArguments(workSpace);
   CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

   CudnnFrontend::Execute("cudnnFindConvolutionBackwardDataAlgorithmEx");
   if(CudnnFrontend::Success()){
      dx = CudnnFrontend::GetOutputDevicePointer();
      *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
      *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgoPerf_t>();
   }
   return CudnnFrontend::GetExitCode();
}
#if CUDNN_VERSION < 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
 									      const cudnnFilterDescriptor_t wDesc,
									      const cudnnTensorDescriptor_t dyDesc,
									      const cudnnConvolutionDescriptor_t convDesc,
									      const cudnnTensorDescriptor_t dxDesc,
									      cudnnConvolutionBwdDataPreference_t preference,
									      size_t memoryLimitInBytes,
									      cudnnConvolutionBwdDataAlgo_t *algo){

     CudnnFrontend::Prepare();
    
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
     CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataPreference_t>(preference);
     CudnnFrontend::AddVariableForArguments<size_t>(memoryLimitInBytes);

     CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithm");
     if(CudnnFrontend::Success()){
          *algo = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgo_t>();
     }
     return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 7000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_t handle,
										 const cudnnFilterDescriptor_t filterDesc,
           								  	 const cudnnTensorDescriptor_t diffDesc,
										 const cudnnConvolutionDescriptor_t convDesc,
										 const cudnnTensorDescriptor_t gradDesc,
										 const int requestedAlgoCount,
										 int *returnedAlgoCount,
										 cudnnConvolutionBwdDataAlgoPerf_t *perfResults){

     CudnnFrontend::Prepare();
   
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)filterDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)diffDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)gradDesc);
     CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);

     CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataAlgorithm_v7");
     if(CudnnFrontend::Success()){
          *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
          *perfResults       = CudnnFrontend::GetOutputVariable<cudnnConvolutionBwdDataAlgoPerf_t>();
     }
     return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
										  const cudnnFilterDescriptor_t wDesc,
										  const cudnnTensorDescriptor_t dyDesc,
										  const cudnnConvolutionDescriptor_t convDesc,
										  const cudnnTensorDescriptor_t dxDesc,
										  cudnnConvolutionBwdDataAlgo_t algo,
										  size_t *sizeInBytes){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataAlgo_t>(algo);
 
    CudnnFrontend::Execute("cudnnGetConvolutionBackwardDataWorkspaceSize");
    if(CudnnFrontend::Success()){
          *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>(); 
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnConvolutionBackwardData(cudnnHandle_t handle,
								  const void *alpha,
								  const cudnnFilterDescriptor_t wDesc,
								  const void *w,
								  const cudnnTensorDescriptor_t dyDesc,
								  const void *dy,
								  const cudnnConvolutionDescriptor_t convDesc,
								  cudnnConvolutionBwdDataAlgo_t algo,
								  void *workSpace,
								  size_t workSpaceSizeInBytes,
								  const void *beta,
								  const cudnnTensorDescriptor_t dxDesc,
								  void *dx){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments(*(float *)alpha);
    //CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddDevicePointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);
    CudnnFrontend::AddVariableForArguments<cudnnConvolutionBwdDataAlgo_t>(algo);
    CudnnFrontend::AddDevicePointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddVariableForArguments(*(float *)beta);
    //CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dx);

    CudnnFrontend::Execute("cudnnConvolutionBackwardData");
    if(CudnnFrontend::Success()){
        dx = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnIm2Col(cudnnHandle_t handle,
						 const cudnnTensorDescriptor_t xDesc,
						 const void *x,
						 const cudnnFilterDescriptor_t wDesc,
						 const cudnnConvolutionDescriptor_t convDesc,
						 void *colBuffer){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)convDesc);

     CudnnFrontend::Execute("cudnnIm2Col");
     if(CudnnFrontend::Success()){
         colBuffer = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSoftmaxForward(cudnnHandle_t handle,
							 cudnnSoftmaxAlgorithm_t algo,
							 cudnnSoftmaxMode_t mode,
							 const void *alpha,
							 const cudnnTensorDescriptor_t xDesc,
							 const void *x,
							 const void *beta,
							 const cudnnTensorDescriptor_t yDesc,
							 void *y){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxAlgorithm_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxMode_t>(mode);
    //CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments(*(float *)alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    //CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments(*(float *)beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);

    CudnnFrontend::Execute("cudnnSoftmaxForward");
    if(CudnnFrontend::Success()){
         y = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSoftmaxBackward(cudnnHandle_t handle,
							  cudnnSoftmaxAlgorithm_t algo,
							  cudnnSoftmaxMode_t mode,
							  const void *alpha,
							  const cudnnTensorDescriptor_t yDesc,
							  const void *y,
							  const cudnnTensorDescriptor_t dyDesc,
							  const void *dy,
							  const void *beta,
							  const cudnnTensorDescriptor_t dxDesc,
							  void *dx){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxAlgorithm_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnSoftmaxMode_t>(mode);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddHostPointerForArguments(dy);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    

    CudnnFrontend::Execute("cudnnSoftmaxBackward");
    if(CudnnFrontend::Success()){
        dx = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t *poolingDesc){
   CudnnFrontend::Prepare();

   CudnnFrontend::Execute("cudnnCreatePoolingDescriptor");
   if(CudnnFrontend::Success())
      *poolingDesc = CudnnFrontend::GetOutputVariable<cudnnPoolingDescriptor_t>();
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                                 cudnnPoolingMode_t mode,
                                                                 cudnnNanPropagation_t maxpoolingNanOpt,
                                                                 int windowHeight,
                                                                 int windowWidth,
                                                                 int verticalPadding,
                                                                 int horizontalPadding,
                                                                 int verticalStride,
                                                                 int horizontalStride){
   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
   CudnnFrontend::AddVariableForArguments<cudnnPoolingMode_t>(mode);
   CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(maxpoolingNanOpt);
   CudnnFrontend::AddVariableForArguments<int>(windowHeight);
   CudnnFrontend::AddVariableForArguments<int>(windowWidth);
   CudnnFrontend::AddVariableForArguments<int>(verticalPadding);
   CudnnFrontend::AddVariableForArguments<int>(horizontalPadding);
   CudnnFrontend::AddVariableForArguments<int>(verticalStride);
   CudnnFrontend::AddVariableForArguments<int>(horizontalStride);

   CudnnFrontend::Execute("cudnnSetPooling2dDescriptor");

   if(CudnnFrontend::Success()){
       poolingDesc = CudnnFrontend::GetOutputVariable<cudnnPoolingDescriptor_t>();
    }  
   return CudnnFrontend::GetExitCode();
}
 
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dDescripto(const cudnnPoolingDescriptor_t poolingDesc,
							        cudnnPoolingMode_t *mode,
								cudnnNanPropagation_t *maxpoolingNanOpt,
								int *windowHeight,
								int *windowWidth,
								int *verticalPadding,
								int *horizontalPadding,
								int *verticalStride,
								int *horizontalStride){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);

    CudnnFrontend::Execute("cudnnGetPooling2dDescripto");
    if(CudnnFrontend::Success()){
        *mode = CudnnFrontend::GetOutputVariable<cudnnPoolingMode_t>();
        *maxpoolingNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
        *windowHeight     = CudnnFrontend::GetOutputVariable<int>();
        *windowWidth      = CudnnFrontend::GetOutputVariable<int>();
        *verticalPadding  = CudnnFrontend::GetOutputVariable<int>();
        *horizontalPadding = CudnnFrontend::GetOutputVariable<int>();
        *verticalStride    = CudnnFrontend::GetOutputVariable<int>();
        *horizontalStride  = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPoolingNdDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                                 const cudnnPoolingMode_t mode,
                                                                 const cudnnNanPropagation_t maxpoolingNanOpt,
                                                                 int nbDims,
                                                                 const int *windowDimA,
                                                                 const int *paddingA,
                                                                 const int *strideA){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddVariableForArguments<cudnnPoolingMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(maxpoolingNanOpt);
    CudnnFrontend::AddVariableForArguments<int>(nbDims);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)windowDimA);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)paddingA);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)strideA);

    CudnnFrontend::Execute("cudnnSetPoolingNdDescriptor");
    if(CudnnFrontend::Success()){
       poolingDesc = CudnnFrontend::GetOutputVariable<cudnnPoolingDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdDescriptor(const cudnnPoolingDescriptor_t poolingDesc,
								 int nbDimsRequested,
								 cudnnPoolingMode_t *mode,
								 cudnnNanPropagation_t *maxpoolingNanOpt,
								 int *nbDims,
								 int *windowDimA,
								 int *paddingA,
								 int *strideA){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
    CudnnFrontend::AddHostPointerForArguments<cudnnNanPropagation_t>((cudnnNanPropagation_t *)maxpoolingNanOpt);
    
    CudnnFrontend::Execute("cudnnGetPoolingNdDescriptor");
    if(CudnnFrontend::Success()){
       *mode = CudnnFrontend::GetOutputVariable<cudnnPoolingMode_t>();
       *nbDims = CudnnFrontend::GetOutputVariable<int>();
       *windowDimA = CudnnFrontend::GetOutputVariable<int>();
       *paddingA   = CudnnFrontend::GetOutputVariable<int>();
       *strideA    = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPoolingNdForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
								       const cudnnTensorDescriptor_t inputTensorDesc,
								       int nbDims,
								       int *outputTensorDimA){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)inputTensorDesc);
     CudnnFrontend::AddVariableForArguments<int>(nbDims);

    CudnnFrontend::Execute("cudnnGetPoolingNdForwardOutputDim");
    if(CudnnFrontend::Success()){
        *outputTensorDimA = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
   								       const cudnnTensorDescriptor_t inputTensorDesc,
								       int *n,
								       int *c,
      								       int *h,
								       int *w){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)inputTensorDesc);
    
    CudnnFrontend::Execute("cudnnGetPooling2dForwardOutputDim");
    if(CudnnFrontend::Success()){
       *n = CudnnFrontend::GetOutputVariable<int>();
       *c = CudnnFrontend::GetOutputVariable<int>();
       *h = CudnnFrontend::GetOutputVariable<int>();
       *w = CudnnFrontend::GetOutputVariable<int>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc){

    CudnnFrontend::Prepare();
    
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);

    CudnnFrontend::Execute("cudnnDestroyPoolingDescriptor");
   
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnPoolingForward(cudnnHandle_t handle,
							 const cudnnPoolingDescriptor_t poolingDesc,
							 const void *alpha,
							 const cudnnTensorDescriptor_t xDesc,
							 const void *x,
							 const void *beta,
							 const cudnnTensorDescriptor_t yDesc,
							 void *y){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddVariableForArguments(*(float *)alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddVariableForArguments(*(float *)beta);
    //CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y); 
    
    CudnnFrontend::Execute("cudnnPoolingForward");
    if(CudnnFrontend::Success()){
       y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnPoolingBackward(cudnnHandle_t handle,
							  const cudnnPoolingDescriptor_t poolingDesc,
							  const void *alpha,
							  const cudnnTensorDescriptor_t yDesc,
							  const void *y,
							  const cudnnTensorDescriptor_t dyDesc,
							  const void *dy,
							  const cudnnTensorDescriptor_t xDesc,
							  const void *x,
							  const void *beta,
							  const cudnnTensorDescriptor_t dxDesc,
							  void *dx){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)poolingDesc);
    CudnnFrontend::AddVariableForArguments(*(float *)alpha);
    //CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddVariableForArguments(*(float *)beta);
    //CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dx);

    CudnnFrontend::Execute("cudnnPoolingBackward");
    if(CudnnFrontend::Success()){
        dx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t *activationDesc){
   CudnnFrontend::Prepare();

   CudnnFrontend::Execute("cudnnCreateActivationDescriptor");
   if(CudnnFrontend::Success())
      *activationDesc = CudnnFrontend::GetOutputVariable<cudnnActivationDescriptor_t>();
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                                                  cudnnActivationMode_t mode,
                                                                  cudnnNanPropagation_t reluNanOpt,
                                                                  double coef){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    CudnnFrontend::AddVariableForArguments<cudnnActivationMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(reluNanOpt);
    CudnnFrontend::AddVariableForArguments<double>(coef);
    CudnnFrontend::Execute("cudnnSetActivationDescriptor");

    if(CudnnFrontend::Success()){
         activationDesc = CudnnFrontend::GetOutputVariable<cudnnActivationDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}
        
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t activationDesc,
                                                                  cudnnActivationMode_t *mode,
                                                                  cudnnNanPropagation_t *reluNanOpt,
                                                                  double *coef){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    CudnnFrontend::Execute("cudnnGetActivationDescriptor");
    if(CudnnFrontend::Success()){
        *mode = CudnnFrontend::GetOutputVariable<cudnnActivationMode_t>();
        *reluNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
        *coef = CudnnFrontend::GetOutputVariable<double>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc){
    CudnnFrontend::Prepare();
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    CudnnFrontend::Execute("cudnnDestroyActivationDescriptor");
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnActivationForward(cudnnHandle_t handle,
							    cudnnActivationDescriptor_t activationDesc,
							    const void *alpha,
							    const cudnnTensorDescriptor_t xDesc,
							    const void *x,
						            const void *beta,
							    const cudnnTensorDescriptor_t yDesc,
							    void *y){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    //CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments(*(float *)alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddVariableForArguments(*(float *)beta);
    //CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);

    CudnnFrontend::Execute("cudnnActivationForward");
    if(CudnnFrontend::Success()){
       y = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(cudnnHandle_t handle,
                                                             cudnnActivationDescriptor_t activationDesc,
							     const void *alpha,
							     const cudnnTensorDescriptor_t yDesc,
							     const void *y,
							     const cudnnTensorDescriptor_t dyDesc,
							     const void *dy,
							     const cudnnTensorDescriptor_t xDesc,
							     const void *x,
							     const void *beta,
						             const cudnnTensorDescriptor_t dxDesc,
							     void *dx){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    //CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddVariableForArguments(*(float *)alpha);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddDevicePointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddDevicePointerForArguments(dy);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddDevicePointerForArguments(x);
    CudnnFrontend::AddVariableForArguments(*(float *)beta);
    //CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    CudnnFrontend::AddDevicePointerForArguments(dx);
    

    CudnnFrontend::Execute("cudnnActivationBackward");
    if(CudnnFrontend::Success()){
       dx = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateLRNDescriptor(cudnnLRNDescriptor_t *normDesc){
    
    CudnnFrontend::Prepare();
   
    CudnnFrontend::Execute("cudnnCreateLRNDescriptor");
    if(CudnnFrontend::Success()){
       *normDesc = CudnnFrontend::GetOutputVariable<cudnnLRNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
               						   unsigned lrnN,
							   double lrnAlpha,
							   double lrnBeta,
							   double lrnK){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<unsigned>(lrnN);
    CudnnFrontend::AddVariableForArguments<double>(lrnAlpha);
    CudnnFrontend::AddVariableForArguments<double>(lrnBeta);
    CudnnFrontend::AddVariableForArguments<double>(lrnK);

    CudnnFrontend::Execute("cudnnSetLRNDescriptor");
    if(CudnnFrontend::Success()){
        normDesc = CudnnFrontend::GetOutputVariable<cudnnLRNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetLRNDescriptor(cudnnLRNDescriptor_t normDesc,
						           unsigned *lrnN,
							   double *lrnAlpha,
							   double *lrnBeta,
							   double *lrnK){

   CudnnFrontend::Prepare();

   CudnnFrontend::Execute("cudnnGetLRNDescriptor");
   if(CudnnFrontend::Success()){
       normDesc = CudnnFrontend::GetOutputVariable<cudnnLRNDescriptor_t>();
       *lrnN    = CudnnFrontend::GetOutputVariable<unsigned>();
       *lrnAlpha = CudnnFrontend::GetOutputVariable<double>();
       *lrnBeta  =  CudnnFrontend::GetOutputVariable<double>();
       *lrnK     =  CudnnFrontend::GetOutputVariable<double>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)lrnDesc);
   
    CudnnFrontend::Execute("cudnnDestroyLRNDescriptor");
   
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelForward(cudnnHandle_t handle,
								 cudnnLRNDescriptor_t normDesc,
								 cudnnLRNMode_t lrnMode,
								 const void *alpha,
								 const cudnnTensorDescriptor_t xDesc,
								 const void *x,
								 const void *beta,
								 const cudnnTensorDescriptor_t yDesc,
								 void *y){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)normDesc);
   CudnnFrontend::AddVariableForArguments<cudnnLRNMode_t>(lrnMode);
   CudnnFrontend::AddHostPointerForArguments(alpha);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
   CudnnFrontend::AddHostPointerForArguments(x);
   CudnnFrontend::AddHostPointerForArguments(beta);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
   

   CudnnFrontend::Execute("cudnnLRNCrossChannelForward");
   if(CudnnFrontend::Success()){
      y = CudnnFrontend::GetOutputDevicePointer();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnLRNCrossChannelBackward(cudnnHandle_t handle,
								  cudnnLRNDescriptor_t normDesc,
								  cudnnLRNMode_t lrnMode,
								  const void *alpha,
								  const cudnnTensorDescriptor_t yDesc,
								  const void *y,
								  const cudnnTensorDescriptor_t dyDesc,
							          const void *dy,
								  const cudnnTensorDescriptor_t xDesc,
								  const void *x,
								  const void *beta,
								  const cudnnTensorDescriptor_t dxDesc,
								  void *dx){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)normDesc);
   CudnnFrontend::AddVariableForArguments<cudnnLRNMode_t>(lrnMode);
   CudnnFrontend::AddHostPointerForArguments(alpha);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
   CudnnFrontend::AddHostPointerForArguments(y);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
   CudnnFrontend::AddHostPointerForArguments(dy);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
   CudnnFrontend::AddHostPointerForArguments(x);
   CudnnFrontend::AddHostPointerForArguments(beta);
   

   CudnnFrontend::Execute("cudnnLRNCrossChannelBackward");
   if(CudnnFrontend::Success()){
        dx     = CudnnFrontend::GetOutputDevicePointer();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationForward(cudnnHandle_t handle,
								       cudnnLRNDescriptor_t normDesc,
								       cudnnDivNormMode_t mode,
								       const void *alpha,
								       const cudnnTensorDescriptor_t xDesc,
								       const void *x,
								       const void *means,
								       void *temp,
								       void *temp2,
								       const void *beta,
								       const cudnnTensorDescriptor_t yDesc,
								       void *y){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)normDesc);
   CudnnFrontend::AddHostPointerForArguments(alpha);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
   CudnnFrontend::AddHostPointerForArguments(x);
   CudnnFrontend::AddHostPointerForArguments(means);
   CudnnFrontend::AddHostPointerForArguments(temp);
   CudnnFrontend::AddHostPointerForArguments(temp2);
   CudnnFrontend::AddHostPointerForArguments(beta);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
   
   CudnnFrontend::Execute("cudnnDivisiveNormalizationForward");
   if(CudnnFrontend::Success()){
       y = CudnnFrontend::GetOutputDevicePointer();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDivisiveNormalizationBackward(cudnnHandle_t handle,
									cudnnLRNDescriptor_t normDesc,
									cudnnDivNormMode_t mode,
									const void *alpha,
									const cudnnTensorDescriptor_t xDesc,
									const void *x,
									const void *means,
									const void *dy,
									void *temp,
									void *temp2,
									const void *beta,
									const cudnnTensorDescriptor_t dXdMeansDesc,
									void *dx,
									void *dMeans){


   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)normDesc);
   CudnnFrontend::AddVariableForArguments<cudnnDivNormMode_t>(mode);
   CudnnFrontend::AddHostPointerForArguments(alpha);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
   CudnnFrontend::AddHostPointerForArguments(x);
   CudnnFrontend::AddHostPointerForArguments(means);
   CudnnFrontend::AddHostPointerForArguments(dy);
   CudnnFrontend::AddHostPointerForArguments(temp);
   CudnnFrontend::AddHostPointerForArguments(temp2);
   CudnnFrontend::AddHostPointerForArguments(beta);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dXdMeansDesc);

   CudnnFrontend::Execute("cudnnDivisiveNormalizationBackward");
   if(CudnnFrontend::Success()){
       dx = CudnnFrontend::GetOutputDevicePointer();
       dMeans = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t derivedBnDesc,
								  const cudnnTensorDescriptor_t xDesc,
								  cudnnBatchNormMode_t mode){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
   CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);

   CudnnFrontend::Execute("cudnnDeriveBNTensorDescriptor");
   if(CudnnFrontend::Success()){
       derivedBnDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(cudnnHandle_t handle,
											      cudnnBatchNormMode_t mode,
											      cudnnBatchNormOps_t bnOps,
											      const cudnnTensorDescriptor_t xDesc,
											      const cudnnTensorDescriptor_t zDesc,
											      const cudnnTensorDescriptor_t yDesc,
									                      const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
											      const cudnnActivationDescriptor_t activationDesc,
											      size_t *sizeInBytes){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)zDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)bnScaleBiasMeanVarDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    
    CudnnFrontend::Execute("cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize");
    if(CudnnFrontend::Success()){
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationBackwardExWorkspaceSize(cudnnHandle_t handle,
										       cudnnBatchNormMode_t mode,
										       cudnnBatchNormOps_t bnOps,
										       const cudnnTensorDescriptor_t xDesc,
										       const cudnnTensorDescriptor_t yDesc,
										       const cudnnTensorDescriptor_t dyDesc,
										       const cudnnTensorDescriptor_t dzDesc,
										       const cudnnTensorDescriptor_t dxDesc,
										       const cudnnTensorDescriptor_t dBnScaleBiasDesc,
										       const cudnnActivationDescriptor_t activationDesc,
										       size_t *sizeInBytes){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dzDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dBnScaleBiasDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
    
    CudnnFrontend::Execute("cudnnGetBatchNormalizationBackwardExWorkspaceSize");
    if(CudnnFrontend::Success()){
       *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetBatchNormalizationTrainingExReserveSpaceSize(cudnnHandle_t handle,
											  cudnnBatchNormMode_t mode,
											  cudnnBatchNormOps_t bnOps,
											  const cudnnActivationDescriptor_t activationDesc,
											  const cudnnTensorDescriptor_t xDesc,
											  size_t *sizeInBytes){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
   CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
   

   CudnnFrontend::Execute("cudnnGetBatchNormalizationTrainingExReserveSpaceSize");
   if(CudnnFrontend::Success()){
      *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(cudnnHandle_t handle,
									    cudnnBatchNormMode_t mode,
									    const void *alpha,
									    const void *beta,
									    const cudnnTensorDescriptor_t xDesc,
									    const void *x,
									    const cudnnTensorDescriptor_t yDesc,
									    void *y,
  									    const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
									    const void *bnScale,
									    const void *bnBias,
									    double exponentialAverageFactor,
									    void *resultRunningMean,
									    void *resultRunningVariance,
								            double epsilon,
									    void *resultSaveMean,
									    void *resultSaveInvVariance){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
   CudnnFrontend::AddHostPointerForArguments(alpha);
   CudnnFrontend::AddHostPointerForArguments(beta);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
   CudnnFrontend::AddHostPointerForArguments(x);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
   CudnnFrontend::AddHostPointerForArguments(y);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)bnScaleBiasMeanVarDesc);
   CudnnFrontend::AddHostPointerForArguments(bnScale);
   CudnnFrontend::AddHostPointerForArguments(bnBias);
   CudnnFrontend::AddVariableForArguments<double>(exponentialAverageFactor);
   CudnnFrontend::AddHostPointerForArguments(resultRunningMean);
   CudnnFrontend::AddHostPointerForArguments(resultRunningVariance);
   CudnnFrontend::AddVariableForArguments<double>(epsilon);

   CudnnFrontend::Execute("cudnnBatchNormalizationForwardTraining");
   if(CudnnFrontend::Success()){
       resultRunningMean = CudnnFrontend::GetOutputDevicePointer();
       resultRunningVariance = CudnnFrontend::GetOutputDevicePointer();
       resultSaveMean        = CudnnFrontend::GetOutputDevicePointer();
       resultSaveInvVariance = CudnnFrontend::GetOutputDevicePointer();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTrainingEx(cudnnHandle_t handle,
									      cudnnBatchNormMode_t mode,
									      cudnnBatchNormOps_t bnOps,
									      const void *alpha,
									      const void *beta,
									      const cudnnTensorDescriptor_t xDesc,
									      const void *xData,
									      const cudnnTensorDescriptor_t zDesc,
									      const void *zData,
									      const cudnnTensorDescriptor_t yDesc,
									      void *yData,
									      const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
									      const void *bnScale,
									      const void *bnBias,
									      double exponentialAverageFactor,
									      void *resultRunningMean,
									      void *resultRunningVariance,
									      double epsilon,
									      void *resultSaveMean,
									      void *resultSaveInvVariance,
									      cudnnActivationDescriptor_t activationDesc,
									      void *workspace,
									      size_t workSpaceSizeInBytes,
									      void *reserveSpace,
									      size_t reserveSpaceSizeInBytes){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
     CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
     CudnnFrontend::AddHostPointerForArguments(alpha);
     CudnnFrontend::AddHostPointerForArguments(beta);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(xData);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)zDesc);
     CudnnFrontend::AddHostPointerForArguments(zData);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)bnScaleBiasMeanVarDesc);
     CudnnFrontend::AddHostPointerForArguments(bnScale);
     CudnnFrontend::AddHostPointerForArguments(bnBias);
     CudnnFrontend::AddVariableForArguments<double>(exponentialAverageFactor);
     CudnnFrontend::AddHostPointerForArguments(resultRunningMean);
     CudnnFrontend::AddHostPointerForArguments(resultRunningVariance);
     CudnnFrontend::AddVariableForArguments<double>(epsilon);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
     CudnnFrontend::AddHostPointerForArguments(workspace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddHostPointerForArguments(reserveSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);


     CudnnFrontend::Execute("cudnnBatchNormalizationForwardTrainingEx");
     if(CudnnFrontend::Success()){
          resultRunningMean     = CudnnFrontend::GetOutputDevicePointer();
          resultRunningVariance = CudnnFrontend::GetOutputDevicePointer();
          resultRunningMean     = CudnnFrontend::GetOutputDevicePointer();
          resultRunningVariance = CudnnFrontend::GetOutputDevicePointer();        
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(cudnnHandle_t handle,
									     cudnnBatchNormMode_t mode,
									     const void *alpha,
									     const void *beta,
									     const cudnnTensorDescriptor_t xDesc,
									     const void *x,
									     const cudnnTensorDescriptor_t yDesc,
									     void *y,
									     const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
									     const void *bnScale,
									     const void *bnBias,
									     const void *estimatedMean,
									     const void *estimatedVariance,
									     double epsilon){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
    CudnnFrontend::AddHostPointerForArguments(alpha);
    CudnnFrontend::AddHostPointerForArguments(beta);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddHostPointerForArguments(y);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)bnScaleBiasMeanVarDesc);
    CudnnFrontend::AddHostPointerForArguments(bnScale);
    CudnnFrontend::AddHostPointerForArguments(bnBias);
    CudnnFrontend::AddHostPointerForArguments(estimatedMean);
    CudnnFrontend::AddHostPointerForArguments(estimatedVariance);
    CudnnFrontend::AddVariableForArguments<double>(epsilon);

    CudnnFrontend::Execute("cudnnBatchNormalizationForwardInference");
   
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(cudnnHandle_t handle,
								     cudnnBatchNormMode_t mode,
								     const void *alphaDataDiff,
								     const void *betaDataDiff,
								     const void *alphaParamDiff,
								     const void *betaParamDiff,
								     const cudnnTensorDescriptor_t xDesc,
								     const void *x,
								     const cudnnTensorDescriptor_t dyDesc,
								     const void *dy,
								     const cudnnTensorDescriptor_t dxDesc,
								     void *dx,
								     const cudnnTensorDescriptor_t dBnScaleBiasDesc,
								     const void *bnScale,
								     void *dBnScaleResult,
								     void *dBnBiasResult,
							             double epsilon,
								     const void *savedMean,
								     const void *savedInvVariance){


      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddHostPointerForArguments(alphaDataDiff);
      CudnnFrontend::AddHostPointerForArguments(betaDataDiff);
      CudnnFrontend::AddHostPointerForArguments(alphaParamDiff);
      CudnnFrontend::AddHostPointerForArguments(betaParamDiff);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
      CudnnFrontend::AddHostPointerForArguments(x);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
      CudnnFrontend::AddHostPointerForArguments(dy);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
      CudnnFrontend::AddHostPointerForArguments(dx);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dBnScaleBiasDesc);
      CudnnFrontend::AddHostPointerForArguments(bnScale);
      CudnnFrontend::AddVariableForArguments<double>(epsilon);
      CudnnFrontend::AddHostPointerForArguments(savedMean);
      CudnnFrontend::AddHostPointerForArguments(savedInvVariance);

      CudnnFrontend::Execute("cudnnBatchNormalizationBackward");
      if(CudnnFrontend::Success()){
         dBnScaleResult = CudnnFrontend::GetOutputDevicePointer();
         dBnBiasResult  = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackwardEx(cudnnHandle_t handle,
								       cudnnBatchNormMode_t mode,
								       cudnnBatchNormOps_t bnOps,
								       const void *alphaDataDiff,
								       const void *betaDataDiff,
								       const void *alphaParamDiff,
								       const void *betaParamDiff,
								       const cudnnTensorDescriptor_t xDesc,
								       const void *xData,
								       const cudnnTensorDescriptor_t yDesc,
								       const void *yData,
								       const cudnnTensorDescriptor_t dyDesc,
								       const void *dyData,
								       const cudnnTensorDescriptor_t dzDesc,
               							       void *dzData,
								       const cudnnTensorDescriptor_t dxDesc,
								       void *dxData,
								       const cudnnTensorDescriptor_t dBnScaleBiasDesc,
								       const void *bnScaleData,
   								       const void *bnBiasData,
								       void *dBnScaleData,
								       void *dBnBiasData,
								       double epsilon,
								       const void *savedMean,
								       const void *savedInvVariance,
								       cudnnActivationDescriptor_t activationDesc,
								       void *workSpace,
								       size_t workSpaceSizeInBytes,
								       void *reserveSpace,
								       size_t reserveSpaceSizeInBytes){

      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddVariableForArguments<cudnnBatchNormMode_t>(mode);
      CudnnFrontend::AddVariableForArguments<cudnnBatchNormOps_t>(bnOps);
      CudnnFrontend::AddHostPointerForArguments(alphaDataDiff);
      CudnnFrontend::AddHostPointerForArguments(betaDataDiff);
      CudnnFrontend::AddHostPointerForArguments(alphaParamDiff);
      CudnnFrontend::AddHostPointerForArguments(betaParamDiff);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
      CudnnFrontend::AddHostPointerForArguments(xData);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
      CudnnFrontend::AddHostPointerForArguments(dyData);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dBnScaleBiasDesc);
      CudnnFrontend::AddHostPointerForArguments(bnScaleData);
      CudnnFrontend::AddHostPointerForArguments(bnBiasData);
      CudnnFrontend::AddVariableForArguments<double>(epsilon);
      CudnnFrontend::AddHostPointerForArguments(savedMean);
      CudnnFrontend::AddHostPointerForArguments(savedInvVariance);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)activationDesc);
      CudnnFrontend::AddHostPointerForArguments(workSpace);
      CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
      CudnnFrontend::AddHostPointerForArguments(reserveSpace);
      CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

      CudnnFrontend::Execute("cudnnBatchNormalizationBackwardEx");
      if(CudnnFrontend::Success()){
       dzData = CudnnFrontend::GetOutputDevicePointer();
       dxData = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateSpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t *stDesc){
     
      CudnnFrontend::Prepare();
    
      CudnnFrontend::Execute("cudnnCreateSpatialTransformerDescriptor");
      if(CudnnFrontend::Success()){
          *stDesc = CudnnFrontend::GetOutputVariable<cudnnSpatialTransformerDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetSpatialTransformerNdDescriptor(cudnnSpatialTransformerDescriptor_t stDesc,
									    cudnnSamplerType_t samplerType,
									    cudnnDataType_t dataType,
									    const int nbDims,
									    const int *dimA){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)stDesc);
     CudnnFrontend::AddVariableForArguments<cudnnSamplerType_t>(samplerType);
     CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
     CudnnFrontend::AddVariableForArguments<int>(nbDims);
     CudnnFrontend::AddHostPointerForArguments<int>((int*)dimA);

     CudnnFrontend::Execute("cudnnSetSpatialTransformerNdDescriptor");
     if(CudnnFrontend::Success()){
         stDesc = CudnnFrontend::GetOutputVariable<cudnnSpatialTransformerDescriptor_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroySpatialTransformerDescriptor(cudnnSpatialTransformerDescriptor_t stDesc){

     CudnnFrontend::Prepare();
 
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)stDesc);

     CudnnFrontend::Execute("cudnnDestroySpatialTransformerDescriptor");
   
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorForward(cudnnHandle_t handle,
									const cudnnSpatialTransformerDescriptor_t stDesc,
									const void *theta,
									void *grid){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)stDesc);
    CudnnFrontend::AddHostPointerForArguments(theta);
    
    CudnnFrontend::Execute("cudnnSpatialTfGridGeneratorForward");
    if(CudnnFrontend::Success()){
        grid = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSpatialTfGridGeneratorBackward(cudnnHandle_t handle,
									 const cudnnSpatialTransformerDescriptor_t stDesc,
									 const void *dgrid,
									 void *dtheta){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)stDesc);
    CudnnFrontend::AddHostPointerForArguments(dgrid);
    
    CudnnFrontend::Execute("cudnnSpatialTfGridGeneratorBackward");
    if(CudnnFrontend::Success()){
        dtheta = CudnnFrontend::GetOutputDevicePointer();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerForward(cudnnHandle_t handle,
								  cudnnSpatialTransformerDescriptor_t stDesc,
								  const void *alpha,
								  const cudnnTensorDescriptor_t xDesc,
								  const void *x,
								  const void *grid,
								  const void *beta,
								  cudnnTensorDescriptor_t yDesc,
								  void *y){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)stDesc);
   CudnnFrontend::AddHostPointerForArguments(alpha);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
   CudnnFrontend::AddHostPointerForArguments(x);
   CudnnFrontend::AddHostPointerForArguments(grid);
   CudnnFrontend::AddHostPointerForArguments(beta);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
   
   CudnnFrontend::Execute("cudnnSpatialTfSamplerForward");
   if(CudnnFrontend::Success()){
     y = CudnnFrontend::GetOutputDevicePointer();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSpatialTfSamplerBackward(cudnnHandle_t handle,
								   cudnnSpatialTransformerDescriptor_t stDesc,
								   const void *alpha,
								   const cudnnTensorDescriptor_t xDesc,
								   const void *x,
								   const void *beta,
								   const cudnnTensorDescriptor_t dxDesc,
								   void *dx,
								   const void *alphaDgrid,
								   const cudnnTensorDescriptor_t dyDesc,
								   const void *dy,
								   const void *grid,
								   const void *betaDgrid,
								   void *dgrid){

  CudnnFrontend::Prepare();

  CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
  CudnnFrontend::AddVariableForArguments<long long int>((long long int)stDesc);
  CudnnFrontend::AddHostPointerForArguments(alpha);
  CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
  CudnnFrontend::AddHostPointerForArguments(x);
  CudnnFrontend::AddHostPointerForArguments(beta);
  CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
  CudnnFrontend::AddHostPointerForArguments(alphaDgrid);
  CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
  CudnnFrontend::AddHostPointerForArguments(dy);
  CudnnFrontend::AddHostPointerForArguments(grid);
  CudnnFrontend::AddHostPointerForArguments(betaDgrid);
  
  CudnnFrontend::Execute("cudnnSpatialTfSamplerBackward");
  if(CudnnFrontend::Success()){
     dx = CudnnFrontend::GetOutputDevicePointer();
     dgrid = CudnnFrontend::GetOutputDevicePointer();
  }
  return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t *dropoutDesc){
  
  CudnnFrontend::Prepare();

  CudnnFrontend::Execute("cudnnCreateDropoutDescriptor");
  if(CudnnFrontend::Success()){
     *dropoutDesc =  CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
  }
  return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc){

  CudnnFrontend::Prepare();

  CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);

  CudnnFrontend::Execute("cudnnDestroyDropoutDescriptor");

  return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t *sizeInBytes){

  CudnnFrontend::Prepare();

  CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
  
  CudnnFrontend::Execute("cudnnDropoutGetStatesSize");
  if(CudnnFrontend::Success()){
     *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
  }
  return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xDesc, size_t *sizeInBytes){

  CudnnFrontend::Prepare();

  CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
  
  CudnnFrontend::Execute("cudnnDropoutGetReserveSpaceSize");
  if(CudnnFrontend::Success()){
     *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
  }
  return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
							       cudnnHandle_t handle,
							       float dropout,
							       void *states,
							       size_t stateSizeInBytes,
							       unsigned long long seed){

  CudnnFrontend::Prepare();

  CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
  CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
  CudnnFrontend::AddVariableForArguments<float>(dropout);
  CudnnFrontend::AddVariableForArguments<size_t>(stateSizeInBytes);
  CudnnFrontend::AddVariableForArguments<unsigned long long>(seed);

  CudnnFrontend::Execute("cudnnSetDropoutDescriptor");
  if(CudnnFrontend::Success()){
      dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
      states      = CudnnFrontend::GetOutputDevicePointer();
  }
  return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRestoreDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
								   cudnnHandle_t handle,
								   float dropout,
								   void *states,
								   size_t stateSizeInBytes,
								   unsigned long long seed){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<float>(dropout);
   CudnnFrontend::AddHostPointerForArguments(states);
   CudnnFrontend::AddVariableForArguments<size_t>(stateSizeInBytes);
   CudnnFrontend::AddVariableForArguments<unsigned long long>(seed);

   CudnnFrontend::Execute("cudnnRestoreDropoutDescriptor");
   if(CudnnFrontend::Success()){
       dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
							      cudnnHandle_t handle,
							      float *dropout,
							      void **states,
						     	      unsigned long long *seed){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   
   CudnnFrontend::Execute("cudnnGetDropoutDescriptor");
   if(CudnnFrontend::Success()){
       *dropout = CudnnFrontend::GetOutputVariable<float>();
       *states  = CudnnFrontend::GetOutputDevicePointer();
       *seed    = CudnnFrontend::GetOutputVariable<unsigned long long>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutForward(cudnnHandle_t handle,
							const cudnnDropoutDescriptor_t dropoutDesc,
							const cudnnTensorDescriptor_t xdesc,
							const void *x,
							const cudnnTensorDescriptor_t ydesc,
							void *y,
							void *reserveSpace,
							size_t reserveSpaceSizeInBytes){
   
   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)xdesc);
   CudnnFrontend::AddHostPointerForArguments(x);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)ydesc);
   CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes); 

   CudnnFrontend::Execute("cudnnDropoutForward");
   if(CudnnFrontend::Success()){
        y = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDropoutBackward(cudnnHandle_t handle,
              						 const cudnnDropoutDescriptor_t dropoutDesc,
							 const cudnnTensorDescriptor_t dydesc,
							 const void *dy,
							 const cudnnTensorDescriptor_t dxdesc,
							 void *dx,
							 void *reserveSpace,
							 size_t reserveSpaceSizeInBytes){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dydesc);
   CudnnFrontend::AddHostPointerForArguments(dy);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxdesc);
   CudnnFrontend::AddHostPointerForArguments(reserveSpace);
   CudnnFrontend::AddVariableForArguments<size_t>((reserveSpaceSizeInBytes));

   CudnnFrontend::Execute("cudnnDropoutBackward");
   if(CudnnFrontend::Success()){
       dx = CudnnFrontend::GetOutputDevicePointer();
   }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc){

   CudnnFrontend::Prepare();

   CudnnFrontend::Execute("cudnnCreateRNNDescriptor");
   if(CudnnFrontend::Success()){
       *rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc){
   
   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);

   CudnnFrontend::Execute("cudnnDestroyRNNDescriptor");

   return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v5(cudnnRNNDescriptor_t rnnDesc,
                                                              int hiddenSize,
                                                              int numLayers,
                                                              cudnnDropoutDescriptor_t dropoutDesc,
                                                              cudnnRNNInputMode_t inputMode,
                                                              cudnnDirectionMode_t direction,
                                                              cudnnRNNMode_t mode,
                                                              cudnnDataType_t mathPrec){



    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(hiddenSize);
    CudnnFrontend::AddVariableForArguments<int>(numLayers);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
    CudnnFrontend::AddVariableForArguments<cudnnRNNInputMode_t>(inputMode);
    CudnnFrontend::AddVariableForArguments<cudnnDirectionMode_t>(direction);
    CudnnFrontend::AddVariableForArguments<cudnnRNNMode_t>(mode);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(mathPrec);

    CudnnFrontend::Execute("cudnnSetRNNDescriptor_v5");
    if(CudnnFrontend::Success()){
        rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 6000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
							  cudnnRNNDescriptor_t rnnDesc,
							  const int hiddenSize,
							  const int numLayers,
							  cudnnDropoutDescriptor_t dropoutDesc,
							  cudnnRNNInputMode_t inputMode,
							  cudnnDirectionMode_t direction,
							  cudnnRNNMode_t mode,
							  cudnnRNNAlgo_t algo,
							  cudnnDataType_t mathPrec){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
   CudnnFrontend::AddVariableForArguments<int>(hiddenSize);
   CudnnFrontend::AddVariableForArguments<int>(numLayers);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
   CudnnFrontend::AddVariableForArguments<cudnnRNNInputMode_t>(inputMode);
   CudnnFrontend::AddVariableForArguments<cudnnDirectionMode_t>(direction);
   CudnnFrontend::AddVariableForArguments<cudnnRNNMode_t>(mode);
   CudnnFrontend::AddVariableForArguments<cudnnRNNAlgo_t>(algo);
   CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(mathPrec);

   CudnnFrontend::Execute("cudnnSetRNNDescriptor_v6");
   if(CudnnFrontend::Success()){
       rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor_v6(cudnnHandle_t handle,
							   cudnnRNNDescriptor_t rnnDesc,
							   int *hiddenSize,
							   int *numLayers,
							   cudnnDropoutDescriptor_t *dropoutDesc,
							   cudnnRNNInputMode_t *inputMode,
							   cudnnDirectionMode_t *direction,
						           cudnnRNNMode_t *mode,
							   cudnnRNNAlgo_t *algo,
							   cudnnDataType_t *mathPrec){
   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);

   CudnnFrontend::Execute("cudnnGetRNNDescriptor_v6");
   if(CudnnFrontend::Success()){
       *hiddenSize = CudnnFrontend::GetOutputVariable<int>();
       *numLayers  = CudnnFrontend::GetOutputVariable<int>();
       *dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
       *inputMode   = CudnnFrontend::GetOutputVariable<cudnnRNNInputMode_t>();
       *direction   = CudnnFrontend::GetOutputVariable<cudnnDirectionMode_t>();
       *mode        = CudnnFrontend::GetOutputVariable<cudnnRNNMode_t>();
       *algo        = CudnnFrontend::GetOutputVariable<cudnnRNNAlgo_t>();
       *mathPrec    = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}
#endif

#if CUDNN_VERSION >= 8000
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v8(cudnnRNNDescriptor_t rnnDesc,
                                                              cudnnRNNAlgo_t algo,
                                                              cudnnRNNMode_t cellMode,
                                                              cudnnRNNBiasMode_t biasMode,
                                                              cudnnDirectionMode_t dirMode,
                                                              cudnnRNNInputMode_t inputMode,
                                                              cudnnDataType_t dataType,
                                                              cudnnDataType_t mathPrec,
                                                              cudnnMathType_t mathType,
                                                              int32_t inputSize,
                                                              int32_t hiddenSize,
                                                              int32_t projSize,
                                                              int32_t numLayers,
                                                              cudnnDropoutDescriptor_t dropoutDesc,
                                                              uint32_t auxFlags){

    CudnnFrontend::Prepare();


    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnRNNAlgo_t>(algo);
    CudnnFrontend::AddVariableForArguments<cudnnRNNMode_t>(cellMode);
    CudnnFrontend::AddVariableForArguments<cudnnRNNBiasMode_t>(biasMode);
    CudnnFrontend::AddVariableForArguments<cudnnDirectionMode_t>(dirMode);
    CudnnFrontend::AddVariableForArguments<cudnnRNNInputMode_t>(inputMode);

    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(mathPrec);
    CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mathType);

    CudnnFrontend::AddVariableForArguments<int>(inputSize);
    CudnnFrontend::AddVariableForArguments<int>(hiddenSize);
    CudnnFrontend::AddVariableForArguments<int>(projSize);
    CudnnFrontend::AddVariableForArguments<int>(numLayers);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
    CudnnFrontend::AddVariableForArguments<int>(auxFlags);

    CudnnFrontend::Execute("cudnnSetRNNDescriptor_v8");
    if(CudnnFrontend::Success()){
        rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDescriptor_v8(cudnnRNNDescriptor_t rnnDesc,
                                                              cudnnRNNAlgo_t *algo,
                                                              cudnnRNNMode_t *cellMode,
                                                              cudnnRNNBiasMode_t *biasMode,
                                                              cudnnDirectionMode_t *dirMode,
                                                              cudnnRNNInputMode_t *inputMode,
                                                              cudnnDataType_t *dataType,
                                                              cudnnDataType_t *mathPrec,
                                                              cudnnMathType_t *mathType,
                                                              int32_t *inputSize,
                                                              int32_t *hiddenSize,
                                                              int32_t *projSize,
                                                              int32_t *numLayers,
                                                              cudnnDropoutDescriptor_t *dropoutDesc,
                                                              uint32_t *auxFlags){
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);

    CudnnFrontend::Execute("cudnnGetRNNDescriptor_v8");
    if(CudnnFrontend::Success()){

        *algo        = CudnnFrontend::GetOutputVariable<cudnnRNNAlgo_t>();
        *cellMode    = CudnnFrontend::GetOutputVariable<cudnnRNNMode_t>();
        *biasMode    = CudnnFrontend::GetOutputVariable<cudnnRNNBiasMode_t>();
        *dirMode     = CudnnFrontend::GetOutputVariable<cudnnDirectionMode_t>();
        *inputMode   = CudnnFrontend::GetOutputVariable<cudnnRNNInputMode_t>();

        *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *mathPrec = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
        *mathType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();

        *inputSize = CudnnFrontend::GetOutputVariable<int>();
        *hiddenSize = CudnnFrontend::GetOutputVariable<int>();
        *projSize = CudnnFrontend::GetOutputVariable<int>();
        *numLayers  = CudnnFrontend::GetOutputVariable<int>();
        *dropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
        *auxFlags  = CudnnFrontend::GetOutputVariable<int>();

    }
    return CudnnFrontend::GetExitCode();
}
#endif
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t mType){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mType);

    CudnnFrontend::Execute("cudnnSetRNNMatrixMathType");

    return CudnnFrontend::GetExitCode();
}      

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t rnnDesc, cudnnMathType_t *mType){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
   
   CudnnFrontend::Execute("cudnnGetRNNMatrixMathType");
   if(CudnnFrontend::Success()){
       *mType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t biasMode){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
   CudnnFrontend::AddVariableForArguments<cudnnRNNBiasMode_t>((biasMode));

   CudnnFrontend::Execute("cudnnSetRNNBiasMode");
   if(CudnnFrontend::Success()){
       rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNBiasMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNBiasMode_t *biasMode){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
   
   CudnnFrontend::Execute("cudnnGetRNNBiasMode");
   if(CudnnFrontend::Success()){
       *biasMode = CudnnFrontend::GetOutputVariable<cudnnRNNBiasMode_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip(cudnnHandle_t handle,
				                     cudnnRNNDescriptor_t rnnDesc,
            				             cudnnRNNClipMode_t clipMode,
                				     cudnnNanPropagation_t clipNanOpt,
                				     double lclip,
                				     double rclip){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
   CudnnFrontend::AddVariableForArguments<cudnnRNNClipMode_t>(clipMode);
   CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(clipNanOpt);
   CudnnFrontend::AddVariableForArguments<double>(lclip);
   CudnnFrontend::AddVariableForArguments<double>(rclip);

   CudnnFrontend::Execute("cudnnRNNSetClip");
  
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNGetClip(cudnnHandle_t handle,
                				     cudnnRNNDescriptor_t rnnDesc,
                				     cudnnRNNClipMode_t *clipMode,
                				     cudnnNanPropagation_t *clipNanOpt,
                      				     double *lclip,
                				     double *rclip){


   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);

   CudnnFrontend::Execute("cudnnRNNGetClip");
   if(CudnnFrontend::Success()){
      *clipMode = CudnnFrontend::GetOutputVariable<cudnnRNNClipMode_t>();
      *clipNanOpt = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
      *lclip      = CudnnFrontend::GetOutputVariable<double>();
      *rclip      = CudnnFrontend::GetOutputVariable<double>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNProjectionLayers(cudnnHandle_t handle,
                            					 cudnnRNNDescriptor_t rnnDesc,
                            					 const int recProjSize,
                            					 const int outProjSize){

      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
      CudnnFrontend::AddVariableForArguments<int>(recProjSize);
      CudnnFrontend::AddVariableForArguments<int>(outProjSize);

      CudnnFrontend::Execute("cudnnSetRNNProjectionLayers");
     
      return CudnnFrontend::GetExitCode();

}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNProjectionLayers(cudnnHandle_t handle,
                            					const cudnnRNNDescriptor_t rnnDesc,
                            					int *recProjSize,
                            					int *outProjSize){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);

     CudnnFrontend::Execute("cudnnGetRNNProjectionLayers");
     if(CudnnFrontend::Success()){
         *recProjSize = CudnnFrontend::GetOutputVariable<int>();
         *outProjSize = CudnnFrontend::GetOutputVariable<int>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreatePersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc,
                             					  const int minibatch,
                             					  const cudnnDataType_t dataType,
                             					  cudnnPersistentRNNPlan_t *plan){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(minibatch);
    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);

    CudnnFrontend::Execute("cudnnCreatePersistentRNNPlan");
    if(CudnnFrontend::Success()){
         *plan = CudnnFrontend::GetOutputVariable<cudnnPersistentRNNPlan_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyPersistentRNNPlan(cudnnPersistentRNNPlan_t plan){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)plan);

    CudnnFrontend::Execute("cudnnDestroyPersistentRNNPlan");
   
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetPersistentRNNPlan(cudnnRNNDescriptor_t rnnDesc, cudnnPersistentRNNPlan_t plan){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)plan);

    CudnnFrontend::Execute("cudnnSetPersistentRNNPlan");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNWorkspaceSize(cudnnHandle_t handle,
                         				     const cudnnRNNDescriptor_t rnnDesc,
                         				     const int seqLength,
                         				     const cudnnTensorDescriptor_t *xDesc,
                         				     size_t *sizeInBytes){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddHostPointerForArguments<cudnnTensorDescriptor_t>((cudnnTensorDescriptor_t*)xDesc);
    
    CudnnFrontend::Execute("cudnnGetRNNWorkspaceSize");
    if(CudnnFrontend::Success()){
       *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(cudnnHandle_t handle,
    					                            const cudnnRNNDescriptor_t rnnDesc,
                               					    const int seqLength,
                               					    const cudnnTensorDescriptor_t *xDesc,
                               					    size_t *sizeInBytes){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddHostPointerForArguments<cudnnTensorDescriptor_t>((cudnnTensorDescriptor_t*)xDesc);

    CudnnFrontend::Execute("cudnnGetRNNTrainingReserveSize");
    if(CudnnFrontend::Success()){
        *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle,
                      					  const cudnnRNNDescriptor_t rnnDesc,
                      					  const cudnnTensorDescriptor_t xDesc,
                      					  size_t *sizeInBytes,
                      					  cudnnDataType_t dataType){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);

     CudnnFrontend::Execute("cudnnGetRNNParamsSize");
     if(CudnnFrontend::Success()){
          *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t handle,
                                				     const cudnnRNNDescriptor_t rnnDesc,
                                   				     const int pseudoLayer,
                                				     const cudnnTensorDescriptor_t xDesc,
                                  				     const cudnnFilterDescriptor_t wDesc,
                                				     const void *w,
                                				     const int linLayerID,
                                				     cudnnFilterDescriptor_t linLayerMatDesc,
                                				     void **linLayerMat){

    CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<int>(pseudoLayer);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddHostPointerForArguments(w);
     CudnnFrontend::AddVariableForArguments<int>(linLayerID);
     
     CudnnFrontend::Execute("cudnnGetRNNLinLayerMatrixParams");
     if(CudnnFrontend::Success()){
          linLayerMatDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
          *linLayerMat    = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams(cudnnHandle_t handle,
                              					   const cudnnRNNDescriptor_t rnnDesc,
                               					   const int pseudoLayer,
                              					   const cudnnTensorDescriptor_t xDesc,
                              					   const cudnnFilterDescriptor_t wDesc,
                              					   const void *w,
	                              				   const int linLayerID,
	                               				   cudnnFilterDescriptor_t linLayerBiasDesc,
                              					   void **linLayerBias){

     CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(pseudoLayer);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<int>(linLayerID);

    CudnnFrontend::Execute("cudnnGetRNNLinLayerBiasParams");
    if(CudnnFrontend::Success()){
          linLayerBiasDesc = CudnnFrontend::GetOutputVariable<cudnnFilterDescriptor_t>();
          *linLayerBias    = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}


extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInference(cudnnHandle_t handle,
                         				      const cudnnRNNDescriptor_t rnnDesc,
                         				      const int seqLength,
                         				      const cudnnTensorDescriptor_t *xDesc,
	                           			      const void *x,
                         				      const cudnnTensorDescriptor_t hxDesc,
                         				      const void *hx,
                         				      const cudnnTensorDescriptor_t cxDesc,
                         				      const void *cx,
                            				      const cudnnFilterDescriptor_t wDesc,
                          				      const void *w,
                         				      const cudnnTensorDescriptor_t *yDesc,
                         				      void *y,
                         				      const cudnnTensorDescriptor_t hyDesc,
                         				      void *hy,
                         				      const cudnnTensorDescriptor_t cyDesc,
                         				      void *cy,
                         				      void *workspace,
                         				      size_t workSpaceSizeInBytes){

     CudnnFrontend::Prepare();
    
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<int>(seqLength);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);
     CudnnFrontend::AddHostPointerForArguments(cx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddHostPointerForArguments(w);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cyDesc);
     CudnnFrontend::AddHostPointerForArguments(workspace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

     CudnnFrontend::Execute("cudnnRNNForwardInference");
     if(CudnnFrontend::Success()){
         y = CudnnFrontend::GetOutputDevicePointer();
         hy  = CudnnFrontend::GetOutputDevicePointer();
         cy  = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();    
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(cudnnHandle_t handle,
                        				     const cudnnRNNDescriptor_t rnnDesc,
                        				     const int seqLength,
                        				     const cudnnTensorDescriptor_t *xDesc,
                        				     const void *x,
                        				     const cudnnTensorDescriptor_t hxDesc,
                        				     const void *hx,
                        				     const cudnnTensorDescriptor_t cxDesc,
                        				     const void *cx,
                        				     const cudnnFilterDescriptor_t wDesc,
                        				     const void *w,
                        				     const cudnnTensorDescriptor_t *yDesc,
                        				     void *y,
                        				     const cudnnTensorDescriptor_t hyDesc,
                        				     void *hy,
                        				     const cudnnTensorDescriptor_t cyDesc,
                        				     void *cy,
                        				     void *workspace,
                        				     size_t workSpaceSizeInBytes,
                        				     void *reserveSpace,
                        				     size_t reserveSpaceSizeInBytes){

      CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<int>(seqLength);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
     CudnnFrontend::AddHostPointerForArguments(hx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);
     CudnnFrontend::AddHostPointerForArguments(cx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddHostPointerForArguments(w);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cyDesc);
     CudnnFrontend::AddHostPointerForArguments(workspace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddHostPointerForArguments(workspace);
     CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

     CudnnFrontend::Execute("cudnnRNNForwardTraining");
     if(CudnnFrontend::Success()){
         y = CudnnFrontend::GetOutputDevicePointer();
         hy  = CudnnFrontend::GetOutputDevicePointer();
         cy  = CudnnFrontend::GetOutputDevicePointer();
         reserveSpace = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}
    
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData(cudnnHandle_t handle,
                     					  const cudnnRNNDescriptor_t rnnDesc,
                     					  const int seqLength,
                     					  const cudnnTensorDescriptor_t *yDesc,
                     					  const void *y,
                     					  const cudnnTensorDescriptor_t *dyDesc,
                     					  const void *dy,
                     					  const cudnnTensorDescriptor_t dhyDesc,
                     					  const void *dhy,
                     					  const cudnnTensorDescriptor_t dcyDesc,
                     					  const void *dcy,
                     					  const cudnnFilterDescriptor_t wDesc,
                     					  const void *w,
                     					  const cudnnTensorDescriptor_t hxDesc,
                     					  const void *hx,
                     					  const cudnnTensorDescriptor_t cxDesc,
                     					  const void *cx,
                     					  const cudnnTensorDescriptor_t *dxDesc,
                     					  void *dx,
                     					  const cudnnTensorDescriptor_t dhxDesc,
                     					  void *dhx,
                     					  const cudnnTensorDescriptor_t dcxDesc,
                     					  void *dcx,
                     					  void *workspace,
                     					  size_t workSpaceSizeInBytes,
                     					  void *reserveSpace,
                     					  size_t reserveSpaceSizeInBytes){


       CudnnFrontend::Prepare();

       CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
       CudnnFrontend::AddVariableForArguments<int>(seqLength);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
       CudnnFrontend::AddHostPointerForArguments(y);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
       CudnnFrontend::AddHostPointerForArguments(dy);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)dhyDesc);
       CudnnFrontend::AddHostPointerForArguments(dhy);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)dcyDesc);
       CudnnFrontend::AddHostPointerForArguments(dcy);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
       CudnnFrontend::AddHostPointerForArguments(w);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
       CudnnFrontend::AddHostPointerForArguments(hx);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);
       CudnnFrontend::AddHostPointerForArguments(cx);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)dhxDesc);
       CudnnFrontend::AddVariableForArguments<long long int>((long long int)dcxDesc);
       CudnnFrontend::AddHostPointerForArguments(workspace);
       CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
       CudnnFrontend::AddHostPointerForArguments(workspace);
       CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
      
       CudnnFrontend::Execute("cudnnRNNBackwardData");
       if(CudnnFrontend::Success()){
          dx = CudnnFrontend::GetOutputDevicePointer();
          dhx  = CudnnFrontend::GetOutputDevicePointer();
          dcx  = CudnnFrontend::GetOutputDevicePointer();
          reserveSpace = CudnnFrontend::GetOutputDevicePointer();
       }
       return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(cudnnHandle_t handle,
                        				     const cudnnRNNDescriptor_t rnnDesc,
                        				     const int seqLength,
                        				     const cudnnTensorDescriptor_t *xDesc,
                        				     const void *x,
                        				     const cudnnTensorDescriptor_t hxDesc,
                        				     const void *hx,
                        				     const cudnnTensorDescriptor_t *yDesc,
                        				     const void *y,
                        				     const void *workspace,
                        				     size_t workSpaceSizeInBytes,
                        				     const cudnnFilterDescriptor_t dwDesc,
                        				     void *dw,
                        				     const void *reserveSpace,
                        				     size_t reserveSpaceSizeInBytes){
     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<int>(seqLength);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
     CudnnFrontend::AddHostPointerForArguments(hx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddHostPointerForArguments(y);
     CudnnFrontend::AddHostPointerForArguments(workspace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
     CudnnFrontend::AddHostPointerForArguments(dw);
     CudnnFrontend::AddHostPointerForArguments(reserveSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
      
     CudnnFrontend::Execute("cudnnRNNBackwardWeights");
      if(CudnnFrontend::Success()){
         dw = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t paddingMode){

      CudnnFrontend::Prepare();
     
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
      CudnnFrontend::AddVariableForArguments<cudnnRNNPaddingMode_t>(paddingMode);

      CudnnFrontend::Execute("cudnnSetRNNPaddingMode");
      if(CudnnFrontend::Success()){
         rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNPaddingMode(cudnnRNNDescriptor_t rnnDesc, cudnnRNNPaddingMode_t *paddingMode){

     CudnnFrontend::Prepare();
     
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
      CudnnFrontend::AddHostPointerForArguments<cudnnRNNPaddingMode_t>((cudnnRNNPaddingMode_t *)paddingMode);

      CudnnFrontend::Execute("cudnnGetRNNPaddingMode");
      if(CudnnFrontend::Success()){
         rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateRNNDataDescriptor(cudnnRNNDataDescriptor_t *rnnDataDesc){

      CudnnFrontend::Prepare();

      CudnnFrontend::Execute("cudnnCreateRNNDataDescriptor");
       if(CudnnFrontend::Success()){
         *rnnDataDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDataDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}
 
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc){


      CudnnFrontend::Prepare();
 
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDataDesc);

      CudnnFrontend::Execute("cudnnDestroyRNNDataDescriptor");

       return CudnnFrontend::GetExitCode();

}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          				      cudnnDataType_t dataType,
                          				      cudnnRNNDataLayout_t layout,
                          				      int maxSeqLength,
                          				      int batchSize,
                          				      int vectorSize,
                          				      const int *seqLengthArray,
                          				      void *paddingFill){


      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDataDesc);
      CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
      CudnnFrontend::AddVariableForArguments<cudnnRNNDataLayout_t>(layout);
      CudnnFrontend::AddVariableForArguments<int>(maxSeqLength);
      CudnnFrontend::AddVariableForArguments<int>(batchSize);
      CudnnFrontend::AddVariableForArguments<int>(vectorSize);
      CudnnFrontend::AddHostPointerForArguments<int>((int*)seqLengthArray);
      CudnnFrontend::AddHostPointerForArguments(paddingFill);

       CudnnFrontend::Execute("cudnnSetRNNDataDescriptor");
       if(CudnnFrontend::Success()){
         rnnDataDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDataDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNDataDescriptor(cudnnRNNDataDescriptor_t rnnDataDesc,
                          				       cudnnDataType_t *dataType,
                          				       cudnnRNNDataLayout_t *layout,
                          				       int *maxSeqLength,
                          				       int *batchSize,
                          				       int *vectorSize,
                          				       int arrayLengthRequested,
                          				       int *seqLengthArray,
                          				       void *paddingFill){


     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDataDesc);
     CudnnFrontend::AddVariableForArguments<int>(arrayLengthRequested);

      CudnnFrontend::Execute("cudnnGetRNNDataDescriptor");
       if(CudnnFrontend::Success()){
            *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
            *layout   = CudnnFrontend::GetOutputVariable<cudnnRNNDataLayout_t>();
            *maxSeqLength = CudnnFrontend::GetOutputVariable<int>();
            *batchSize    = CudnnFrontend::GetOutputVariable<int>();
            *vectorSize   = CudnnFrontend::GetOutputVariable<int>();
            *seqLengthArray = CudnnFrontend::GetOutputVariable<int>();
            paddingFill    = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTrainingEx(cudnnHandle_t handle,
                                                               const cudnnRNNDescriptor_t rnnDesc,
                          				       const cudnnRNNDataDescriptor_t xDesc,
                          				       const void *x,
                          				       const cudnnTensorDescriptor_t hxDesc,
                          				       const void *hx,
                          				       const cudnnTensorDescriptor_t cxDesc,
                          				       const void *cx,
                          				       const cudnnFilterDescriptor_t wDesc,
                          				       const void *w,
                          				       const cudnnRNNDataDescriptor_t yDesc,
                          				       void *y,
                          				       const cudnnTensorDescriptor_t hyDesc,
                          				       void *hy,
                          				       const cudnnTensorDescriptor_t cyDesc,
                          				       void *cy,
                           				       const cudnnRNNDataDescriptor_t kDesc, 
                          				       const void *keys,                     
                          				       const cudnnRNNDataDescriptor_t cDesc, 
                          				       void *cAttn,                          
                          				       const cudnnRNNDataDescriptor_t iDesc, 
                          				       void *iAttn,                          
                          				       const cudnnRNNDataDescriptor_t qDesc, 
                          				       void *queries,                        
                          				       void *workSpace,
                          				       size_t workSpaceSizeInBytes,
                          				       void *reserveSpace,
                          				       size_t reserveSpaceSizeInBytes){



     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc); 
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
     CudnnFrontend::AddHostPointerForArguments(hx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);
     CudnnFrontend::AddHostPointerForArguments(cx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);  
     CudnnFrontend::AddHostPointerForArguments(w);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc); 
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)kDesc);
     CudnnFrontend::AddHostPointerForArguments(keys);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
     CudnnFrontend::AddHostPointerForArguments(cAttn);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)iDesc);
     CudnnFrontend::AddHostPointerForArguments(iAttn);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)qDesc);
     CudnnFrontend::AddHostPointerForArguments(queries);
     CudnnFrontend::AddHostPointerForArguments(workSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddHostPointerForArguments(reserveSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
         
     CudnnFrontend::Execute("cudnnRNNForwardTrainingEx");
       if(CudnnFrontend::Success()){
            y    = CudnnFrontend::GetOutputDevicePointer();
            hy   = CudnnFrontend::GetOutputDevicePointer();
            cy   = CudnnFrontend::GetOutputDevicePointer();
            reserveSpace = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();     
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardInferenceEx(cudnnHandle_t handle,
                           				        const cudnnRNNDescriptor_t rnnDesc,
                           					const cudnnRNNDataDescriptor_t xDesc,
                           					const void *x,
                           					const cudnnTensorDescriptor_t hxDesc,
                           					const void *hx,
                           					const cudnnTensorDescriptor_t cxDesc,
                           					const void *cx,
                           					const cudnnFilterDescriptor_t wDesc,
                           					const void *w,
                           					const cudnnRNNDataDescriptor_t yDesc,
                           					void *y,
                           					const cudnnTensorDescriptor_t hyDesc,
                           					void *hy,
                           					const cudnnTensorDescriptor_t cyDesc,
                           					void *cy,
                           					const cudnnRNNDataDescriptor_t kDesc,  
                           					const void *keys,                      
                           					const cudnnRNNDataDescriptor_t cDesc, 
                           					void *cAttn,                          
                           					const cudnnRNNDataDescriptor_t iDesc, 
                           					void *iAttn,                          
                           					const cudnnRNNDataDescriptor_t qDesc, 
                           					void *queries,                        
                           					void *workSpace,
                           					size_t workSpaceSizeInBytes){


     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
     CudnnFrontend::AddHostPointerForArguments(hx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);
     CudnnFrontend::AddHostPointerForArguments(cx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddHostPointerForArguments(w);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)kDesc);
     CudnnFrontend::AddHostPointerForArguments(keys);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cDesc);
     CudnnFrontend::AddHostPointerForArguments(cAttn);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)iDesc);
     CudnnFrontend::AddHostPointerForArguments(iAttn);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)qDesc);
     CudnnFrontend::AddHostPointerForArguments(queries);
     CudnnFrontend::AddHostPointerForArguments(workSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

      CudnnFrontend::Execute("cudnnRNNForwardInferenceEx");
       if(CudnnFrontend::Success()){
            y    = CudnnFrontend::GetOutputDevicePointer();
            hy   = CudnnFrontend::GetOutputDevicePointer();
            cy   = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}
    
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardDataEx(cudnnHandle_t handle,
                       					    const cudnnRNNDescriptor_t rnnDesc,
                       					    const cudnnRNNDataDescriptor_t yDesc,
                       					    const void *y,
                       					    const cudnnRNNDataDescriptor_t dyDesc,
                       					    const void *dy,
                       					    const cudnnRNNDataDescriptor_t dcDesc, 
                       				            const void *dcAttn,                    
                       					    const cudnnTensorDescriptor_t dhyDesc,
                       					    const void *dhy,
                       				            const cudnnTensorDescriptor_t dcyDesc,
                       					    const void *dcy,
                       					    const cudnnFilterDescriptor_t wDesc,
                       					    const void *w,
                       					    const cudnnTensorDescriptor_t hxDesc,
                       					    const void *hx,
                       					    const cudnnTensorDescriptor_t cxDesc,
                       					    const void *cx,
                       					    const cudnnRNNDataDescriptor_t dxDesc,
                       				            void *dx,
                       					    const cudnnTensorDescriptor_t dhxDesc,
                       					    void *dhx,
                       					    const cudnnTensorDescriptor_t dcxDesc,
                       					    void *dcx,
                       					    const cudnnRNNDataDescriptor_t dkDesc, 
                       					    void *dkeys,                           
                       					    void *workSpace,
                       					    size_t workSpaceSizeInBytes,
                       					    void *reserveSpace,
 	 	 			                    size_t reserveSpaceSizeInBytes){

    
      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
      CudnnFrontend::AddHostPointerForArguments(y);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);
      CudnnFrontend::AddHostPointerForArguments(dy);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dcDesc); 
      CudnnFrontend::AddHostPointerForArguments(dcAttn);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dhyDesc);
      CudnnFrontend::AddHostPointerForArguments(dhy);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dcyDesc);
      CudnnFrontend::AddHostPointerForArguments(dcy);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);   
      CudnnFrontend::AddHostPointerForArguments(w);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
      CudnnFrontend::AddHostPointerForArguments(hx);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);
      CudnnFrontend::AddHostPointerForArguments(cx);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dhxDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dcxDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dkDesc);
      CudnnFrontend::AddHostPointerForArguments(dkeys);
      CudnnFrontend::AddHostPointerForArguments(workSpace);
      CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
      CudnnFrontend::AddHostPointerForArguments(reserveSpace);
      CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

      
      CudnnFrontend::Execute("cudnnRNNBackwardDataEx");
       if(CudnnFrontend::Success()){
            dx    = CudnnFrontend::GetOutputDevicePointer();
            dhx   = CudnnFrontend::GetOutputDevicePointer();
            dcx   = CudnnFrontend::GetOutputDevicePointer();
            reserveSpace = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeightsEx(cudnnHandle_t handle,
                          				       const cudnnRNNDescriptor_t rnnDesc,
                          				       const cudnnRNNDataDescriptor_t xDesc,
                          				       const void *x,
                          				       const cudnnTensorDescriptor_t hxDesc,
                          				       const void *hx,
                          				       const cudnnRNNDataDescriptor_t yDesc,
                          				       const void *y,
                          				       void *workSpace,
                          				       size_t workSpaceSizeInBytes,
                          				       const cudnnFilterDescriptor_t dwDesc,
                          				       void *dw,
                          				       void *reserveSpace,
                          				       size_t reserveSpaceSizeInBytes){



     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
     CudnnFrontend::AddHostPointerForArguments(hx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddHostPointerForArguments(y);
     CudnnFrontend::AddHostPointerForArguments(workSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
     CudnnFrontend::AddHostPointerForArguments(dw);
     CudnnFrontend::AddHostPointerForArguments(reserveSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

      CudnnFrontend::Execute("cudnnRNNBackwardWeightsEx");
       if(CudnnFrontend::Success()){
            dw    = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNAlgorithmDescriptor(cudnnHandle_t handle, cudnnRNNDescriptor_t rnnDesc, cudnnAlgorithmDescriptor_t algoDesc){

     CudnnFrontend::Prepare();


     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoDesc);
     
      CudnnFrontend::Execute("cudnnSetRNNAlgorithmDescriptor");
      
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count){

     CudnnFrontend::Prepare();
   
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    
    CudnnFrontend::Execute("cudnnGetRNNForwardInferenceAlgorithmMaxCount");
    if(CudnnFrontend::Success()){
	  *count = CudnnFrontend::GetOutputVariable<int>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardInferenceAlgorithmEx(cudnnHandle_t handle,
                                        				     const cudnnRNNDescriptor_t rnnDesc,
                                        				     const int seqLength,
                                        				     const cudnnTensorDescriptor_t *xDesc,
                                        				     const void *x,
                                        				     const cudnnTensorDescriptor_t hxDesc,
                                        				     const void *hx,
                                        				     const cudnnTensorDescriptor_t cxDesc,
                                        				     const void *cx,
                                        				     const cudnnFilterDescriptor_t wDesc,
                                        				     const void *w,
                                        				     const cudnnTensorDescriptor_t *yDesc,
                                        				     void *y,
                                        				     const cudnnTensorDescriptor_t hyDesc,
                                        				     void *hy,
                                        				     const cudnnTensorDescriptor_t cyDesc,
                                        				     void *cy,
                                        				     const float findIntensity,
                                        				     const int requestedAlgoCount,
                                        				     int *returnedAlgoCount,
                                        				     cudnnAlgorithmPerformance_t *perfResults,
                                        				     void *workspace,
                                        				     size_t workSpaceSizeInBytes){


    CudnnFrontend::Prepare();


    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    CudnnFrontend::AddVariableForArguments<int>(seqLength);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
    CudnnFrontend::AddHostPointerForArguments(x);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
    CudnnFrontend::AddHostPointerForArguments(hx);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);
    CudnnFrontend::AddHostPointerForArguments(cx);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
    CudnnFrontend::AddHostPointerForArguments(w);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)hyDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)cyDesc);
    CudnnFrontend::AddVariableForArguments<float>(findIntensity); 
    CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
    CudnnFrontend::AddHostPointerForArguments(workspace);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

     CudnnFrontend::Execute("cudnnFindRNNForwardInferenceAlgorithmEx");
    if(CudnnFrontend::Success()){
         y = CudnnFrontend::GetOutputDevicePointer();
         hy = CudnnFrontend::GetOutputDevicePointer();
         cy = CudnnFrontend::GetOutputDevicePointer();
         *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
         *perfResults = CudnnFrontend::GetOutputVariable<cudnnAlgorithmPerformance_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardTrainingAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count){

    CudnnFrontend::Prepare();
   
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
    
    CudnnFrontend::Execute("cudnnGetRNNForwardTrainingAlgorithmMaxCount");
    if(CudnnFrontend::Success()){
         *count = CudnnFrontend::GetOutputVariable<int>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindRNNForwardTrainingAlgorithmEx(cudnnHandle_t handle,
                                       				           const cudnnRNNDescriptor_t rnnDesc,
                                       				           const int seqLength,
                                       				           const cudnnTensorDescriptor_t *xDesc,
                                       					   const void *x,
                                       					   const cudnnTensorDescriptor_t hxDesc,
                                       					   const void *hx,
                                       					   const cudnnTensorDescriptor_t cxDesc,
                                       					   const void *cx,
                                       					   const cudnnFilterDescriptor_t wDesc,
                                       					   const void *w,
                                       					   const cudnnTensorDescriptor_t *yDesc,
                                       					   void *y,
                                       					   const cudnnTensorDescriptor_t hyDesc,
                                       					   void *hy,
                                       					   const cudnnTensorDescriptor_t cyDesc,
                                       					   void *cy,
                                       					   const float findIntensity,
                                       					   const int requestedAlgoCount,
                                       					   int *returnedAlgoCount,
                                       					   cudnnAlgorithmPerformance_t *perfResults,
                                       				           void *workspace,
                                       					   size_t workSpaceSizeInBytes,
                                       				           void *reserveSpace,
                                       				           size_t reserveSpaceSizeInBytes){


     CudnnFrontend::Prepare();
    
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::AddVariableForArguments<int>(seqLength);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
     CudnnFrontend::AddHostPointerForArguments(x);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
     CudnnFrontend::AddHostPointerForArguments(hx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);
     CudnnFrontend::AddHostPointerForArguments(cx);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);
     CudnnFrontend::AddHostPointerForArguments(w); 
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)hyDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)cyDesc);
     CudnnFrontend::AddVariableForArguments<float>(findIntensity);
     CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
     CudnnFrontend::AddHostPointerForArguments(workspace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
     CudnnFrontend::AddHostPointerForArguments(reserveSpace);
     CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

      CudnnFrontend::Execute("cudnnFindRNNForwardTrainingAlgorithmEx");
    if(CudnnFrontend::Success()){
         y = CudnnFrontend::GetOutputDevicePointer();
         hy = CudnnFrontend::GetOutputDevicePointer();
         cy = CudnnFrontend::GetOutputDevicePointer();
         *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
         *perfResults = CudnnFrontend::GetOutputVariable<cudnnAlgorithmPerformance_t>();
      }
      return CudnnFrontend::GetExitCode();
}
    
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardDataAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count){


     CudnnFrontend::Prepare();


     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     
     CudnnFrontend::Execute("cudnnGetRNNBackwardDataAlgorithmMaxCount");
    if(CudnnFrontend::Success()){
         *count = CudnnFrontend::GetOutputVariable<int>();
      }
      return CudnnFrontend::GetExitCode();
}
    
extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardDataAlgorithmEx(cudnnHandle_t handle,
                                    				        const cudnnRNNDescriptor_t rnnDesc,
                                    					const int seqLength,
                                    					const cudnnTensorDescriptor_t *yDesc,
                                    					const void *y,
                                    					const cudnnTensorDescriptor_t *dyDesc,
                                    					const void *dy,
                                    					const cudnnTensorDescriptor_t dhyDesc,
                                    					const void *dhy,
                                    					const cudnnTensorDescriptor_t dcyDesc,
                                    					const void *dcy,
                                    					const cudnnFilterDescriptor_t wDesc,
                                    					const void *w,
                                    					const cudnnTensorDescriptor_t hxDesc,
                                    					const void *hx,
	 			                                        const cudnnTensorDescriptor_t cxDesc,
                                    					const void *cx,
                                    					const cudnnTensorDescriptor_t *dxDesc,
                                    					void *dx,
                                    					const cudnnTensorDescriptor_t dhxDesc,
                                    					void *dhx,
                                    					const cudnnTensorDescriptor_t dcxDesc,
                                    					void *dcx,
                                    					const float findIntensity,
                                    					const int requestedAlgoCount,
                                    					int *returnedAlgoCount,
                                    					cudnnAlgorithmPerformance_t *perfResults,
                                    					void *workspace,
                                    					size_t workSpaceSizeInBytes,
                                    					void *reserveSpace,
                                    					size_t reserveSpaceSizeInBytes){


     CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
      CudnnFrontend::AddVariableForArguments<int>(seqLength);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
      CudnnFrontend::AddHostPointerForArguments(y);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dyDesc);  
      CudnnFrontend::AddHostPointerForArguments(dy);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dhyDesc);  
      CudnnFrontend::AddHostPointerForArguments(dhy); 
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dcyDesc);  
      CudnnFrontend::AddHostPointerForArguments(dcy);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)wDesc);  
      CudnnFrontend::AddHostPointerForArguments(w);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);  
      CudnnFrontend::AddHostPointerForArguments(hx);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)cxDesc);  
      CudnnFrontend::AddHostPointerForArguments(cx);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dxDesc);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dhxDesc);  
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dcxDesc);
      CudnnFrontend::AddVariableForArguments<float>(findIntensity);
      CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);
      CudnnFrontend::AddHostPointerForArguments(workspace);
      CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
      CudnnFrontend::AddHostPointerForArguments(reserveSpace);
      CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);

      CudnnFrontend::Execute("cudnnFindRNNBackwardDataAlgorithmEx");
      if(CudnnFrontend::Success()){
           dx = CudnnFrontend::GetOutputDevicePointer();
           dhx = CudnnFrontend::GetOutputDevicePointer();
           dcx =  CudnnFrontend::GetOutputDevicePointer();
           *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
           *perfResults       =  CudnnFrontend::GetOutputVariable<cudnnAlgorithmPerformance_t>();
      }
      return CudnnFrontend::GetExitCode();
}
      
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNBackwardWeightsAlgorithmMaxCount(cudnnHandle_t handle, const cudnnRNNDescriptor_t rnnDesc, int *count){
     
      CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
     CudnnFrontend::Execute("cudnnGetRNNBackwardWeightsAlgorithmMaxCount");
      if(CudnnFrontend::Success()){
           *count = CudnnFrontend::GetOutputVariable<int>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFindRNNBackwardWeightsAlgorithmEx(cudnnHandle_t handle,
                                       					    const cudnnRNNDescriptor_t rnnDesc,
                                       					    const int seqLength,
                                       				 	    const cudnnTensorDescriptor_t *xDesc,
                                       					    const void *x,
                                       					    const cudnnTensorDescriptor_t hxDesc,
                                       					    const void *hx,
                                       					    const cudnnTensorDescriptor_t *yDesc,
                                       					    const void *y,
                                       					    const float findIntensity,
                                       					    const int requestedAlgoCount,
                                       					    int *returnedAlgoCount,
                                       					    cudnnAlgorithmPerformance_t *perfResults,
                                       					    const void *workspace,
                                       					    size_t workSpaceSizeInBytes,
                                       					    const cudnnFilterDescriptor_t dwDesc,
                                       					    void *dw,
                                       				            const void *reserveSpace,
                                       					    size_t reserveSpaceSizeInBytes){



     CudnnFrontend::Prepare();


      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
      CudnnFrontend::AddVariableForArguments<int>(seqLength);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)xDesc);
      CudnnFrontend::AddHostPointerForArguments(x);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)hxDesc);
      CudnnFrontend::AddHostPointerForArguments(hx);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)yDesc);
      CudnnFrontend::AddHostPointerForArguments(y);
      CudnnFrontend::AddVariableForArguments<float>(findIntensity);
      CudnnFrontend::AddVariableForArguments<int>(requestedAlgoCount);        
      CudnnFrontend::AddHostPointerForArguments(workspace);
      CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)dwDesc);
      CudnnFrontend::AddHostPointerForArguments(dw);
      CudnnFrontend::AddHostPointerForArguments(reserveSpace);
      CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
  
       CudnnFrontend::Execute("cudnnFindRNNBackwardWeightsAlgorithmEx");
      if(CudnnFrontend::Success()){
           dw = CudnnFrontend::GetOutputDevicePointer();
           *returnedAlgoCount = CudnnFrontend::GetOutputVariable<int>();
           *perfResults       =  CudnnFrontend::GetOutputVariable<cudnnAlgorithmPerformance_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateSeqDataDescriptor(cudnnSeqDataDescriptor_t *seqDataDesc){


      CudnnFrontend::Prepare();

      CudnnFrontend::Execute("cudnnCreateSeqDataDescriptor");
      if(CudnnFrontend::Success()){
           *seqDataDesc       =  CudnnFrontend::GetOutputVariable<cudnnSeqDataDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}
   
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroySeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc){


      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)seqDataDesc);
      
      CudnnFrontend::Execute("cudnnDestroySeqDataDescriptor");
      
      return CudnnFrontend::GetExitCode();    
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetSeqDataDescriptor(cudnnSeqDataDescriptor_t seqDataDesc,
                          				       cudnnDataType_t dataType,
                          				       int nbDims,
                          				       const int *dimA,
                          				       const cudnnSeqDataAxis_t *axes,
                          				       size_t seqLengthArraySize,
                          				       const int *seqLengthArray,
                          				       void *paddingFill){


      CudnnFrontend::Prepare();


      CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
      CudnnFrontend::AddVariableForArguments<int>(nbDims); 
      CudnnFrontend::AddHostPointerForArguments<int>((int*)dimA);
      CudnnFrontend::AddHostPointerForArguments<cudnnSeqDataAxis_t>((cudnnSeqDataAxis_t*)axes);
      CudnnFrontend::AddVariableForArguments<size_t>(seqLengthArraySize); 
      CudnnFrontend::AddHostPointerForArguments<int>((int*)seqLengthArray);
      CudnnFrontend::AddHostPointerForArguments(paddingFill);

      CudnnFrontend::Execute("cudnnSetSeqDataDescriptor");
      if(CudnnFrontend::Success()){
          seqDataDesc = CudnnFrontend::GetOutputVariable<cudnnSeqDataDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetSeqDataDescriptor(const cudnnSeqDataDescriptor_t seqDataDesc,
                          				       cudnnDataType_t *dataType,
                          				       int *nbDims,
                          				       int nbDimsRequested,
	 			                               int *dimA,
                          				       cudnnSeqDataAxis_t *axes,
                          				       size_t *seqLengthArraySize,
                          				       size_t seqLengthSizeRequested,
                          				       int *seqLengthArray,
                          				       void *paddingFill){



     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)seqDataDesc);
     CudnnFrontend::AddVariableForArguments<int>(nbDimsRequested);
     CudnnFrontend::AddVariableForArguments<size_t>(seqLengthSizeRequested);

     CudnnFrontend::Execute("cudnnGetSeqDataDescriptor");
      if(CudnnFrontend::Success()){
          *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
          *nbDims   = CudnnFrontend::GetOutputVariable<int>();
          *dimA     = CudnnFrontend::GetOutputVariable<int>();
          *axes     = CudnnFrontend::GetOutputVariable<cudnnSeqDataAxis_t>();
          *seqLengthArraySize = CudnnFrontend::GetOutputVariable<size_t>();
          *seqLengthArray     = CudnnFrontend::GetOutputVariable<int>();
          paddingFill        = CudnnFrontend::GetOutputDevicePointer();
      }
      return CudnnFrontend::GetExitCode();
}

     
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateAttnDescriptor(cudnnAttnDescriptor_t *attnDesc){

     CudnnFrontend::Prepare();

     CudnnFrontend::Execute("cudnnCreateAttnDescriptor");
      if(CudnnFrontend::Success()){
           *attnDesc = CudnnFrontend::GetOutputVariable<cudnnAttnDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyAttnDescriptor(cudnnAttnDescriptor_t attnDesc){

     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)attnDesc);

     CudnnFrontend::Execute("cudnnDestroyAttnDescriptor");
    
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
                       					    unsigned attnMode,
                       					    int nHeads,
                       					    double smScaler,
                       					    cudnnDataType_t dataType,
                       				  	    cudnnDataType_t computePrec,
                       					    cudnnMathType_t mathType,
                       					    cudnnDropoutDescriptor_t attnDropoutDesc,
                       					    cudnnDropoutDescriptor_t postDropoutDesc,
                       					    int qSize,
                       					    int kSize,
                       					    int vSize,
                       					    int qProjSize,
                       					    int kProjSize,
                       					    int vProjSize,
                       				  	    int oProjSize,
                       					    int qoMaxSeqLength,
                       					    int kvMaxSeqLength,
                       					    int maxBatchSize,
                       					    int maxBeamSize){



     CudnnFrontend::Prepare();



     CudnnFrontend::AddVariableForArguments<unsigned>(attnMode);
     CudnnFrontend::AddVariableForArguments<int>(nHeads);
     CudnnFrontend::AddVariableForArguments<double>(smScaler);
     CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(dataType);
     CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(computePrec);
     CudnnFrontend::AddVariableForArguments<cudnnMathType_t>(mathType);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)attnDropoutDesc);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)postDropoutDesc);
     CudnnFrontend::AddVariableForArguments<int>(qSize);
     CudnnFrontend::AddVariableForArguments<int>(kSize);
     CudnnFrontend::AddVariableForArguments<int>(vSize);
     CudnnFrontend::AddVariableForArguments<int>(qProjSize);
     CudnnFrontend::AddVariableForArguments<int>(kProjSize);
     CudnnFrontend::AddVariableForArguments<int>(vProjSize);
     CudnnFrontend::AddVariableForArguments<int>(oProjSize);
     CudnnFrontend::AddVariableForArguments<int>(qoMaxSeqLength);
     CudnnFrontend::AddVariableForArguments<int>(kvMaxSeqLength);
     CudnnFrontend::AddVariableForArguments<int>(maxBatchSize);
     CudnnFrontend::AddVariableForArguments<int>(maxBeamSize);

      CudnnFrontend::Execute("cudnnSetAttnDescriptor");
      if(CudnnFrontend::Success()){
           attnDesc = CudnnFrontend::GetOutputVariable<cudnnAttnDescriptor_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAttnDescriptor(cudnnAttnDescriptor_t attnDesc,
                       				            unsigned *attnMode,
                       					    int *nHeads,
                       					    double *smScaler,
                       					    cudnnDataType_t *dataType,
                       					    cudnnDataType_t *computePrec,
                       					    cudnnMathType_t *mathType,
                       					    cudnnDropoutDescriptor_t *attnDropoutDesc,
                       					    cudnnDropoutDescriptor_t *postDropoutDesc,
                       					    int *qSize,
                       					    int *kSize,
                       					    int *vSize,
                       					    int *qProjSize,
                       					    int *kProjSize,
                       					    int *vProjSize,
                       				            int *oProjSize,
                       					    int *qoMaxSeqLength,
                       					    int *kvMaxSeqLength,
                       				            int *maxBatchSize,
                       					    int *maxBeamSize){

      CudnnFrontend::Prepare();


      CudnnFrontend::AddVariableForArguments<long long int>((long long int)attnDesc);

      CudnnFrontend::Execute("cudnnGetAttnDescriptor");
      if(CudnnFrontend::Success()){
  	    *attnMode =   CudnnFrontend::GetOutputVariable<unsigned>();
  	    *nHeads = CudnnFrontend::GetOutputVariable<int>();
  	    *smScaler =   CudnnFrontend::GetOutputVariable<double>();
  	    *dataType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
  	    *computePrec = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
  	    *mathType = CudnnFrontend::GetOutputVariable<cudnnMathType_t>();
  	    *attnDropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
  	    *postDropoutDesc = CudnnFrontend::GetOutputVariable<cudnnDropoutDescriptor_t>();
  	    *qSize = CudnnFrontend::GetOutputVariable<int>();
  	    *kSize = CudnnFrontend::GetOutputVariable<int>();
  	    *vSize = CudnnFrontend::GetOutputVariable<int>();
  	    *qProjSize = CudnnFrontend::GetOutputVariable<int>();
  	    *vProjSize = CudnnFrontend::GetOutputVariable<int>();
  	    *oProjSize = CudnnFrontend::GetOutputVariable<int>();
  	    *qoMaxSeqLength = CudnnFrontend::GetOutputVariable<int>();
  	    *kvMaxSeqLength = CudnnFrontend::GetOutputVariable<int>();
  	    *maxBatchSize = CudnnFrontend::GetOutputVariable<int>();
  	    *maxBeamSize = CudnnFrontend::GetOutputVariable<int>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnBuffers(cudnnHandle_t handle,
                             					  const cudnnAttnDescriptor_t attnDesc,
                             					  size_t *weightSizeInBytes,
                             					  size_t *workSpaceSizeInBytes,
                             					  size_t *reserveSpaceSizeInBytes){


     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)attnDesc);
     
     CudnnFrontend::Execute("cudnnGetMultiHeadAttnBuffers");
      if(CudnnFrontend::Success()){
            *weightSizeInBytes       = CudnnFrontend::GetOutputVariable<size_t>();
            *workSpaceSizeInBytes    = CudnnFrontend::GetOutputVariable<size_t>();
            *reserveSpaceSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
      }
      return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetMultiHeadAttnWeights(cudnnHandle_t handle,
                             					  const cudnnAttnDescriptor_t attnDesc,
                             					  cudnnMultiHeadAttnWeightKind_t wKind,
                             					  size_t weightSizeInBytes,
                             					  const void *weights,
                             					  cudnnTensorDescriptor_t wDesc,
                             					  void **wAddr){


     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)attnDesc);
     CudnnFrontend::AddVariableForArguments<cudnnMultiHeadAttnWeightKind_t>(wKind);
     CudnnFrontend::AddVariableForArguments<size_t>(weightSizeInBytes);
     CudnnFrontend::AddHostPointerForArguments(weights);
      
     CudnnFrontend::Execute("cudnnGetMultiHeadAttnWeights");
     if(CudnnFrontend::Success()){
         wDesc = CudnnFrontend::GetOutputVariable<cudnnTensorDescriptor_t>();
        *wAddr =  CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnForward(cudnnHandle_t handle,
                          				     const cudnnAttnDescriptor_t attnDesc,
                          				     int currIdx,
                          				     const int *loWinIdx,
                          				     const int *hiWinIdx,
                          				     const int *seqLengthArrayQRO,
                          				     const int *seqLengthArrayKV,
                          				     const cudnnSeqDataDescriptor_t qDesc,
                          				     const void *queries,
                          				     const void *residuals,
                          				     const cudnnSeqDataDescriptor_t kDesc,
                          				     const void *keys,
                          			             const cudnnSeqDataDescriptor_t vDesc,
			                                     const void *values,
                          				     const cudnnSeqDataDescriptor_t oDesc,
                          				     void *out,
                          				     size_t weightSizeInBytes,
                          				     const void *weights,
 				                             size_t workSpaceSizeInBytes,
                          				     void *workSpace,
                          				     size_t reserveSpaceSizeInBytes,
                          			             void *reserveSpace){



    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)attnDesc);
    CudnnFrontend::AddVariableForArguments<int>(currIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)loWinIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)hiWinIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)seqLengthArrayQRO);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)seqLengthArrayKV);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)qDesc);
    CudnnFrontend::AddHostPointerForArguments(queries);
    CudnnFrontend::AddHostPointerForArguments(residuals);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)kDesc);
    CudnnFrontend::AddHostPointerForArguments(keys);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)vDesc);
    CudnnFrontend::AddHostPointerForArguments(values);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)oDesc);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(weights);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);

     CudnnFrontend::Execute("cudnnMultiHeadAttnForward");
     if(CudnnFrontend::Success()){
        out = CudnnFrontend::GetOutputDevicePointer();
        workSpace = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardData(cudnnHandle_t handle,
                               					    const cudnnAttnDescriptor_t attnDesc,
                               					    const int *loWinIdx,
                               					    const int *hiWinIdx,
                               					    const int *seqLengthArrayDQDO,
                               					    const int *seqLengthArrayDKDV,
                               					    const cudnnSeqDataDescriptor_t doDesc,
                               					    const void *dout,
                               					    const cudnnSeqDataDescriptor_t dqDesc,
                               					    void *dqueries,
                               					    const void *queries,
                               					    const cudnnSeqDataDescriptor_t dkDesc,
                               					    void *dkeys,
                               					    const void *keys,
                               					    const cudnnSeqDataDescriptor_t dvDesc,
                               					    void *dvalues,
                               					    const void *values,
                               					    size_t weightSizeInBytes,
                               					    const void *weights,
                               					    size_t workSpaceSizeInBytes,
                               					    void *workSpace,
                               					    size_t reserveSpaceSizeInBytes,
                               					    void *reserveSpace){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)attnDesc);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)loWinIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)hiWinIdx);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)seqLengthArrayDQDO);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)seqLengthArrayDKDV);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)doDesc);
    CudnnFrontend::AddHostPointerForArguments(dout);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dqDesc);
    CudnnFrontend::AddHostPointerForArguments(queries);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dkDesc);
    CudnnFrontend::AddHostPointerForArguments(keys);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)dvDesc);
    CudnnFrontend::AddHostPointerForArguments(values);
    CudnnFrontend::AddVariableForArguments<size_t>(weightSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(weights);
    CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(workSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
    CudnnFrontend::AddHostPointerForArguments(reserveSpace);

     CudnnFrontend::Execute("cudnnMultiHeadAttnBackwardData");
     if(CudnnFrontend::Success()){
        dout = CudnnFrontend::GetOutputDevicePointer();
        dqueries = CudnnFrontend::GetOutputDevicePointer();
        dqueries = CudnnFrontend::GetOutputDevicePointer();
        dkeys = CudnnFrontend::GetOutputDevicePointer();
        dvalues = CudnnFrontend::GetOutputDevicePointer();
        workSpace = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnMultiHeadAttnBackwardWeights(cudnnHandle_t handle,
                                  				       const cudnnAttnDescriptor_t attnDesc,
                                  				       cudnnWgradMode_t addGrad,
                                  				       const cudnnSeqDataDescriptor_t qDesc,
                                  				       const void *queries,
                                  				       const cudnnSeqDataDescriptor_t kDesc,
                                  				       const void *keys,
                                  				       const cudnnSeqDataDescriptor_t vDesc,
                                  				       const void *values,
                                  				       const cudnnSeqDataDescriptor_t doDesc,
                                  				       const void *dout,
                                  				       size_t weightSizeInBytes,
                                  				       const void *weights,
                                  				       void *dweights,
                                  				       size_t workSpaceSizeInBytes,
                                  				       void *workSpace,
                                  				       size_t reserveSpaceSizeInBytes,
                                  				       void *reserveSpace){


      CudnnFrontend::Prepare();

      CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)attnDesc);
      CudnnFrontend::AddVariableForArguments<cudnnWgradMode_t>(addGrad);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)qDesc);
      CudnnFrontend::AddHostPointerForArguments(queries);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)kDesc);
      CudnnFrontend::AddHostPointerForArguments(keys);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)vDesc);
      CudnnFrontend::AddHostPointerForArguments(values);
      CudnnFrontend::AddVariableForArguments<long long int>((long long int)doDesc);
      CudnnFrontend::AddHostPointerForArguments(dout);
      CudnnFrontend::AddVariableForArguments<size_t>(weightSizeInBytes);
      CudnnFrontend::AddHostPointerForArguments(weights);
      CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);
      CudnnFrontend::AddHostPointerForArguments(workSpace);
      CudnnFrontend::AddVariableForArguments<size_t>(reserveSpaceSizeInBytes);
      CudnnFrontend::AddHostPointerForArguments(reserveSpace);

       CudnnFrontend::Execute("cudnnMultiHeadAttnBackwardWeights");
     if(CudnnFrontend::Success()){
        dweights = CudnnFrontend::GetOutputDevicePointer();
        workSpace = CudnnFrontend::GetOutputDevicePointer();
        reserveSpace = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateCTCLossDescriptor(cudnnCTCLossDescriptor_t *ctcLossDesc){

     CudnnFrontend::Prepare();
     
     CudnnFrontend::Execute("cudnnCreateCTCLossDescriptor");
     if(CudnnFrontend::Success()){
         *ctcLossDesc = CudnnFrontend::GetOutputVariable<cudnnCTCLossDescriptor_t>();
     }
     return CudnnFrontend::GetExitCode();      
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t compType){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(compType);
   
    CudnnFrontend::Execute("cudnnSetCTCLossDescriptor");
    if(CudnnFrontend::Success()){
       ctcLossDesc = CudnnFrontend::GetOutputVariable<cudnnCTCLossDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                                				 cudnnDataType_t compType,
                            					 cudnnLossNormalizationMode_t normMode,
                            					 cudnnNanPropagation_t gradMode){


     CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(compType);
     CudnnFrontend::AddVariableForArguments<cudnnLossNormalizationMode_t>(normMode);
     CudnnFrontend::AddVariableForArguments<cudnnNanPropagation_t>(gradMode);

    CudnnFrontend::Execute("cudnnSetCTCLossDescriptorEx");
    if(CudnnFrontend::Success()){
        ctcLossDesc = CudnnFrontend::GetOutputVariable<cudnnCTCLossDescriptor_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc, cudnnDataType_t *compType){
    
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)ctcLossDesc);

    CudnnFrontend::Execute("cudnnGetCTCLossDescriptor");
    if(CudnnFrontend::Success()){
       *compType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
    }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossDescriptorEx(cudnnCTCLossDescriptor_t ctcLossDesc,
                             					 cudnnDataType_t *compType,
                            					 cudnnLossNormalizationMode_t *normMode,
                            					 cudnnNanPropagation_t *gradMode){


   CudnnFrontend::Prepare();


   CudnnFrontend::AddVariableForArguments<long long int>((long long int)ctcLossDesc);
   
   CudnnFrontend::Execute("cudnnGetCTCLossDescriptorEx");
   if(CudnnFrontend::Success()){
      *compType = CudnnFrontend::GetOutputVariable<cudnnDataType_t>();
      *normMode = CudnnFrontend::GetOutputVariable<cudnnLossNormalizationMode_t>();
      *gradMode = CudnnFrontend::GetOutputVariable<cudnnNanPropagation_t>();
   } 
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyCTCLossDescriptor(cudnnCTCLossDescriptor_t ctcLossDesc){


   CudnnFrontend::Prepare();


   CudnnFrontend::AddVariableForArguments<long long int>((long long int)ctcLossDesc);

   CudnnFrontend::Execute("cudnnDestroyCTCLossDescriptor");

   return CudnnFrontend::GetExitCode();
}

#if CUDNN_VERSION < 8204
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCTCLoss(cudnnHandle_t handle,
    						  const cudnnTensorDescriptor_t
        					  probsDesc,       
    						  const void *probs,  
    						  const int *labels,  
    					  	  const int *labelLengths,                     
    						  const int *inputLengths,                      
    						  void *costs,                                 
    						  const cudnnTensorDescriptor_t gradientsDesc, 
    						  const void *gradients,   
    						  cudnnCTCLossAlgo_t algo, 
    						  cudnnCTCLossDescriptor_t ctcLossDesc,
    						  void *workspace,              
    						  size_t workSpaceSizeInBytes){ 


    CudnnFrontend::Prepare();

     CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)probsDesc);
     CudnnFrontend::AddHostPointerForArguments(probs);
     CudnnFrontend::AddHostPointerForArguments<int>((int*)labels);
     CudnnFrontend::AddHostPointerForArguments<int>((int*)labelLengths);
     CudnnFrontend::AddHostPointerForArguments<int>((int*)inputLengths);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)gradientsDesc);
     CudnnFrontend::AddVariableForArguments<cudnnCTCLossAlgo_t>(algo);
     CudnnFrontend::AddVariableForArguments<long long int>((long long int)ctcLossDesc);
     CudnnFrontend::AddHostPointerForArguments(workspace);
     CudnnFrontend::AddVariableForArguments<size_t>(workSpaceSizeInBytes);

     CudnnFrontend::Execute("cudnnCTCLoss");
     if(CudnnFrontend::Success()){
          costs = CudnnFrontend::GetOutputDevicePointer();
          gradients = CudnnFrontend::GetOutputDevicePointer();
     }
     return CudnnFrontend::GetExitCode();
}
#endif

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCTCLossWorkspaceSize(cudnnHandle_t handle,
    								  const cudnnTensorDescriptor_t probsDesc, 
   	 							  const cudnnTensorDescriptor_t gradientsDesc,
    								  const int *labels,                          
    								  const int *labelLengths,                     
    								  const int *inputLengths,                     
    								  cudnnCTCLossAlgo_t algo,                     
   	 							  cudnnCTCLossDescriptor_t ctcLossDesc,
    								  size_t *sizeInBytes){ 

    
 
    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)probsDesc);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)gradientsDesc);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)labels);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)labelLengths);
    CudnnFrontend::AddHostPointerForArguments<int>((int*)inputLengths);
    CudnnFrontend::AddVariableForArguments<cudnnCTCLossAlgo_t>(algo);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)ctcLossDesc);
    
    CudnnFrontend::Execute("cudnnGetCTCLossWorkspaceSize");
    if(CudnnFrontend::Success()){
         *sizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t *algoDesc){

    CudnnFrontend::Prepare();

    CudnnFrontend::Execute("cudnnCreateAlgorithmDescriptor");
    if(CudnnFrontend::Success()){
        *algoDesc = CudnnFrontend::GetOutputVariable<cudnnAlgorithmDescriptor_t>();
    }   
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t algorithm){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoDesc);
    CudnnFrontend::AddVariableForArguments<cudnnAlgorithm_t>(algorithm);
   
    CudnnFrontend::Execute("cudnnSetAlgorithmDescriptor");
    if(CudnnFrontend::Success()){
          algoDesc = CudnnFrontend::GetOutputVariable<cudnnAlgorithmDescriptor_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t algoDesc, cudnnAlgorithm_t *algorithm){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoDesc);
   CudnnFrontend::AddHostPointerForArguments<cudnnAlgorithm_t>((cudnnAlgorithm_t*)algorithm);

   CudnnFrontend::Execute("cudnnGetAlgorithmDescriptor");
   
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCopyAlgorithmDescriptor(const cudnnAlgorithmDescriptor_t src, cudnnAlgorithmDescriptor_t dest){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)src);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dest); 

   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t algoDesc){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoDesc); 

   CudnnFrontend::Execute("cudnnDestroyAlgorithmDescriptor");

   return CudnnFrontend::GetExitCode(); 
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToCreate){


   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<int>(numberToCreate);

   CudnnFrontend::Execute("cudnnCreateAlgorithmPerformance");
   if(CudnnFrontend::Success()){
        *algoPerf = CudnnFrontend::GetOutputVariable<cudnnAlgorithmPerformance_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetAlgorithmPerformance(cudnnAlgorithmPerformance_t algoPerf,
                             					  cudnnAlgorithmDescriptor_t algoDesc,
                             					  cudnnStatus_t status,
                             					  float time,
                             					  size_t memory){
   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoPerf);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoDesc);
   CudnnFrontend::AddVariableForArguments<cudnnStatus_t>(status);
   CudnnFrontend::AddVariableForArguments<float>(time);
   CudnnFrontend::AddVariableForArguments<size_t>(memory);

   CudnnFrontend::Execute("cudnnSetAlgorithmPerformance");
   if(CudnnFrontend::Success()){
      algoPerf = CudnnFrontend::GetOutputVariable<cudnnAlgorithmPerformance_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmPerformance(const cudnnAlgorithmPerformance_t algoPerf,
                             					  cudnnAlgorithmDescriptor_t *algoDesc,
                             					  cudnnStatus_t *status,
                             					  float *time,
                             					  size_t *memory){



    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoPerf);
    
    CudnnFrontend::Execute("cudnnGetAlgorithmPerformance");
    if(CudnnFrontend::Success()){
        *algoDesc = CudnnFrontend::GetOutputVariable<cudnnAlgorithmDescriptor_t>();
        *status   = CudnnFrontend::GetOutputVariable<cudnnStatus_t>();
        *memory   = CudnnFrontend::GetOutputVariable<size_t>();
     }
     return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyAlgorithmPerformance(cudnnAlgorithmPerformance_t *algoPerf, int numberToDestroy){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddHostPointerForArguments<long long int>((long long int*)algoPerf);
    CudnnFrontend::AddVariableForArguments<int>(numberToDestroy);

    CudnnFrontend::Execute("cudnnDestroyAlgorithmPerformance");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetAlgorithmSpaceSize(cudnnHandle_t handle, cudnnAlgorithmDescriptor_t algoDesc, size_t *algoSpaceSizeInBytes){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoDesc);

   CudnnFrontend::Execute("cudnnGetAlgorithmSpaceSize");
   if(CudnnFrontend::Success()){
       *algoSpaceSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>(); 
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSaveAlgorithm(cudnnHandle_t handle,
                   					cudnnAlgorithmDescriptor_t algoDesc,
                   					void *algoSpace,
                   					size_t algoSpaceSizeInBytes){

    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoDesc);
    CudnnFrontend::AddHostPointerForArguments(algoSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(algoSpaceSizeInBytes);

    CudnnFrontend::Execute("cudnnSaveAlgorithm");
    
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnRestoreAlgorithm(cudnnHandle_t handle,
                      					   void *algoSpace,
                      					   size_t algoSpaceSizeInBytes,
                      					   cudnnAlgorithmDescriptor_t algoDesc){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
    CudnnFrontend::AddHostPointerForArguments(algoSpace);
    CudnnFrontend::AddVariableForArguments<size_t>(algoSpaceSizeInBytes);
    CudnnFrontend::AddVariableForArguments<long long int>((long long int)algoDesc);

    CudnnFrontend::Execute("cudnnRestoreAlgorithm");
    
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetCallback(unsigned mask, void *udata, cudnnCallback_t fptr){


    CudnnFrontend::Prepare();

    CudnnFrontend::AddVariableForArguments<unsigned>(mask);
    CudnnFrontend::AddHostPointerForArguments(udata);
    CudnnFrontend::AddVariableForArguments<cudnnCallback_t>(fptr);

    CudnnFrontend::Execute("cudnnSetCallback");

    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetCallback(unsigned *mask, void **udata, cudnnCallback_t *fptr){


   CudnnFrontend::Prepare();

   CudnnFrontend::Execute("cudnnGetCallback");
   if(CudnnFrontend::Success()){
       *mask = CudnnFrontend::GetOutputVariable<unsigned>();
       *udata = CudnnFrontend::GetOutputDevicePointer();
       *fptr  = CudnnFrontend::GetOutputVariable<cudnnCallback_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t *constPack, cudnnFusedOps_t ops){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddHostPointerForArguments<cudnnFusedOpsConstParamPack_t>((cudnnFusedOpsConstParamPack_t*)constPack);
   CudnnFrontend::AddVariableForArguments<cudnnFusedOps_t>(ops);

   CudnnFrontend::Execute("cudnnCreateFusedOpsConstParamPack");
   
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsConstParamPack(cudnnFusedOpsConstParamPack_t constPack){


   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)constPack);

   CudnnFrontend::Execute("cudnnDestroyFusedOpsConstParamPack");

   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsConstParamPackAttribute(cudnnFusedOpsConstParamPack_t constPack,
                                        				     cudnnFusedOpsConstParamLabel_t paramLabel,
                                        				     const void *param){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)constPack);
   CudnnFrontend::AddVariableForArguments<cudnnFusedOpsConstParamLabel_t>(paramLabel);
   CudnnFrontend::AddHostPointerForArguments(param);

   CudnnFrontend::Execute("cudnnSetFusedOpsConstParamPackAttribute");

   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsConstParamPackAttribute(const cudnnFusedOpsConstParamPack_t constPack,
                                        		   		     cudnnFusedOpsConstParamLabel_t paramLabel,
                                                                             void *param,
                                                                             int *isNULL){


   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)constPack);
   CudnnFrontend::AddVariableForArguments<cudnnFusedOpsConstParamLabel_t>(paramLabel);
   CudnnFrontend::AddHostPointerForArguments(param);
   CudnnFrontend::AddHostPointerForArguments<int>((int*)isNULL);

   CudnnFrontend::Execute("cudnnGetFusedOpsConstParamPackAttribute");
   if(CudnnFrontend::Success()){
         *isNULL = CudnnFrontend::GetOutputVariable<int>();
   }
    return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t *varPack, cudnnFusedOps_t ops){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<cudnnFusedOps_t>(ops);

   CudnnFrontend::Execute("cudnnCreateFusedOpsVariantParamPack");
   if(CudnnFrontend::Success()){
      *varPack = CudnnFrontend::GetOutputVariable<cudnnFusedOpsVariantParamPack_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsVariantParamPack(cudnnFusedOpsVariantParamPack_t varPack){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)varPack);

   CudnnFrontend::Execute("cudnnDestroyFusedOpsVariantParamPack");
   
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetFusedOpsVariantParamPackAttribute(cudnnFusedOpsVariantParamPack_t varPack,
                                          				       cudnnFusedOpsVariantParamLabel_t paramLabel,
                                          				       void *ptr){


   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)varPack);
   CudnnFrontend::AddVariableForArguments<cudnnFusedOpsVariantParamLabel_t>(paramLabel);
   CudnnFrontend::AddHostPointerForArguments(ptr);

   CudnnFrontend::Execute("cudnnSetFusedOpsVariantParamPackAttribute");
   
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetFusedOpsVariantParamPackAttribute(const cudnnFusedOpsVariantParamPack_t varPack,
                                          				       cudnnFusedOpsVariantParamLabel_t paramLabel,
                                          				       void *ptr){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)varPack);
   CudnnFrontend::AddVariableForArguments<cudnnFusedOpsVariantParamLabel_t>(paramLabel);

   CudnnFrontend::Execute("cudnnGetFusedOpsVariantParamPackAttribute");
   if(CudnnFrontend::Success()){
       ptr = CudnnFrontend::GetOutputDevicePointer();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateFusedOpsPlan(cudnnFusedOpsPlan_t *plan, cudnnFusedOps_t ops){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddHostPointerForArguments<cudnnFusedOpsPlan_t>((cudnnFusedOpsPlan_t*)plan);
   CudnnFrontend::AddVariableForArguments<cudnnFusedOps_t>(ops);

   CudnnFrontend::Execute("cudnnCreateFusedOpsPlan");

   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyFusedOpsPlan(cudnnFusedOpsPlan_t plan){


   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)plan);

   CudnnFrontend::Execute("cudnnDestroyFusedOpsPlan");

   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnMakeFusedOpsPlan(cudnnHandle_t handle,
                      					   cudnnFusedOpsPlan_t plan,
                      					   const cudnnFusedOpsConstParamPack_t constPack,
                      					   size_t *workspaceSizeInBytes){

   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)plan);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)constPack);

   CudnnFrontend::Execute("cudnnMakeFusedOpsPlan");
   if(CudnnFrontend::Success()){
       *workspaceSizeInBytes = CudnnFrontend::GetOutputVariable<size_t>();
   }
   return CudnnFrontend::GetExitCode();
}

extern "C" cudnnStatus_t CUDNNWINAPI cudnnFusedOpsExecute(cudnnHandle_t handle, const cudnnFusedOpsPlan_t plan, cudnnFusedOpsVariantParamPack_t varPack){

  CudnnFrontend::Prepare();

  CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
  CudnnFrontend::AddVariableForArguments<long long int>((long long int)plan);
  CudnnFrontend::AddVariableForArguments<long long int>((long long int)varPack);

  CudnnFrontend::Execute("cudnnFusedOpsExecute");
  
  return CudnnFrontend::GetExitCode();
}
/*
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(cudnnHandle_t handle,
                         				      cudnnRNNDescriptor_t rnnDesc,
                         				      const int hiddenSize,
                         				      const int numLayers,
                         				      cudnnDropoutDescriptor_t dropoutDesc,
                         			  	      cudnnRNNInputMode_t inputMode,
                         				      cudnnDirectionMode_t direction,
                         				      cudnnRNNMode_t mode,
                         				      cudnnRNNAlgo_t algo,
                         				      cudnnDataType_t mathPrec){


   CudnnFrontend::Prepare();

   CudnnFrontend::AddVariableForArguments<long long int>((long long int)handle);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)rnnDesc);
   CudnnFrontend::AddVariableForArguments<int>(hiddenSize);
   CudnnFrontend::AddVariableForArguments<int>(numLayers);
   CudnnFrontend::AddVariableForArguments<long long int>((long long int)dropoutDesc);
   CudnnFrontend::AddVariableForArguments<cudnnRNNInputMode_t>(inputMode);
   CudnnFrontend::AddVariableForArguments<cudnnDirectionMode_t>(direction);
   CudnnFrontend::AddVariableForArguments<cudnnRNNMode_t>(mode);
   CudnnFrontend::AddVariableForArguments<cudnnRNNAlgo_t>(algo);
   CudnnFrontend::AddVariableForArguments<cudnnDataType_t>(mathPrec);

   CudnnFrontend::Execute("cudnnSetRNNDescriptor_v6");
   if(CudnnFrontend::Success()){
       rnnDesc = CudnnFrontend::GetOutputVariable<cudnnRNNDescriptor_t>();
   }
   return CudnnFrontend::GetExitCode();
}
*/


