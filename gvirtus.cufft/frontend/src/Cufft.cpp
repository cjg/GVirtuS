/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2011  The University of Napoli Parthenope at Naples.
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
 * @file   Cufft.cpp
 * 
 * @author Giuseppe Coviello <giuseppe.coviello@uniparthenope.it>
 * @author Vincenzo Santopietro <vincenzo.santopietro@uniparthenope.it>
 * @date   Sat Oct 10 10:51:58 2009
 *
 * @brief
 */

#include <cufft.h>
#include <cufftXt.h>

#include "Frontend.h"
#include "CufftFrontend.h"

using namespace std;

extern "C" cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type,
        int batch) {
    CufftFrontend::Prepare();
    CufftFrontend::AddHostPointerForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments(nx);
    CufftFrontend::AddVariableForArguments(type);
    CufftFrontend::AddVariableForArguments(batch);
    
    CufftFrontend::Execute("cufftPlan1d");
    if(CufftFrontend::Success())
        *plan = *(CufftFrontend::GetOutputHostPointer<cufftHandle>());
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny,
        cufftType type) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments(nx);
    CufftFrontend::AddVariableForArguments(ny);
    CufftFrontend::AddVariableForArguments(type);
    
    CufftFrontend::Execute("cufftPlan2d");
    if(CufftFrontend::Success())
        *plan = *(CufftFrontend::GetOutputHostPointer<cufftHandle>());
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz,
        cufftType type) {
    CufftFrontend::Prepare();
    CufftFrontend::AddHostPointerForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments(nx);
    CufftFrontend::AddVariableForArguments(ny);
    CufftFrontend::AddVariableForArguments(nz);
    CufftFrontend::AddVariableForArguments(type);
    CufftFrontend::Execute("cufftPlan3d");
    if(CufftFrontend::Success())
        *plan =  *(CufftFrontend::GetOutputHostPointer<cufftHandle>());
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t * workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    CufftFrontend::Execute("cufftEstimate1d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t * workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<int>(ny);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    CufftFrontend::Execute("cufftEstimate2d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t * workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<int>(ny);
    CufftFrontend::AddVariableForArguments<int>(nz);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    CufftFrontend::Execute("cufftEstimate3d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftEstimateMany( int rank, int *n, 
        int *inembed, int istride, int idist, int *onembed, int ostride,
        int odist, cufftType type, int batch,size_t * workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<int>(rank);
    CufftFrontend::AddHostPointerForArguments<int>(n);
    CufftFrontend::AddHostPointerForArguments<int>(inembed);
    CufftFrontend::AddVariableForArguments<int>(istride);
    CufftFrontend::AddVariableForArguments<int>(idist);
    CufftFrontend::AddHostPointerForArguments<int>(onembed);
    CufftFrontend::AddVariableForArguments<int>(ostride);
    CufftFrontend::AddVariableForArguments<int>(odist);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    CufftFrontend::Execute("cufftEstimateMany");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}


extern "C" cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch, size_t * workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    
    CufftFrontend::Execute("cufftMakePlan1d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftMakePlan2d(cufftHandle plan, int nx,int ny, cufftType type, size_t * workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<int>(ny);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    
    CufftFrontend::Execute("cufftMakePlan2d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}


extern "C" cufftResult cufftMakePlan3d(cufftHandle plan, int nx,int ny,int nz, cufftType type, size_t * workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<int>(ny);
    CufftFrontend::AddVariableForArguments<int>(nz);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    
    CufftFrontend::Execute("cufftMakePlan3d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftMakePlanMany(cufftHandle plan, int rank, int *n, 
        int *inembed, int istride, int idist, int *onembed, int ostride,
        int odist, cufftType type, int batch,size_t * workSize) {
    CufftFrontend::Prepare();
    //Passing arguments
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(rank);
    CufftFrontend::AddHostPointerForArguments<int>(n);
    CufftFrontend::AddHostPointerForArguments<int>(inembed);
    CufftFrontend::AddVariableForArguments<int>(istride);
    CufftFrontend::AddVariableForArguments<int>(idist);
    CufftFrontend::AddHostPointerForArguments<int>(onembed);
    CufftFrontend::AddVariableForArguments<int>(ostride);
    CufftFrontend::AddVariableForArguments<int>(odist);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    
    CufftFrontend::Execute("cufftMakePlanMany");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

#if CUDART_VERSION >= 7000
extern "C" cufftResult cufftMakePlanMany64(cufftHandle plan, int rank, long long int *n, 
        long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride,
        long long int odist, cufftType type, long long int batch,size_t * workSize) {
    CufftFrontend::Prepare();
    //Passing arguments
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(rank);
    CufftFrontend::AddHostPointerForArguments<long long int>(n);
    CufftFrontend::AddHostPointerForArguments<long long int>(inembed);
    CufftFrontend::AddVariableForArguments<long long int>(istride);
    CufftFrontend::AddVariableForArguments<long long int>(idist);
    CufftFrontend::AddHostPointerForArguments<long long int>(onembed);
    CufftFrontend::AddVariableForArguments<long long int>(ostride);
    CufftFrontend::AddVariableForArguments<long long int>(odist);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<long long int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    
    CufftFrontend::Execute("cufftMakePlanMany64");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}
#endif

extern "C" cufftResult cufftGetSize1d(cufftHandle handle, int nx, cufftType type,
        int batch, size_t *workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<cufftHandle>(handle);
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
                                   
    CufftFrontend::Execute("cufftGetSize1d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftGetSize2d(cufftHandle handle, int nx, int ny, cufftType type,
        size_t *workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<cufftHandle>(handle);
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<int>(ny);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
                                   
    CufftFrontend::Execute("cufftGetSize2d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}


extern "C" cufftResult cufftGetSize3d(cufftHandle handle, int nx, int ny, int nz, cufftType type,
        size_t *workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<cufftHandle>(handle);
    CufftFrontend::AddVariableForArguments<int>(nx);
    CufftFrontend::AddVariableForArguments<int>(ny);
    CufftFrontend::AddVariableForArguments<int>(nz);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
                                   
    CufftFrontend::Execute("cufftGetSize3d");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftGetSizeMany(cufftHandle handle, int rank, int *n, 
        int *inembed, int istride, int idist, int *onembed, int ostride,
        int odist, cufftType type, int batch, size_t *workSize){
    CufftFrontend::Prepare();
    
    CufftFrontend::AddVariableForArguments<cufftHandle>(handle);
    CufftFrontend::AddVariableForArguments<int>(rank);
    CufftFrontend::AddHostPointerForArguments<int>(n);
    CufftFrontend::AddHostPointerForArguments<int>(inembed);
    CufftFrontend::AddVariableForArguments<int>(istride);
    CufftFrontend::AddVariableForArguments<int>(idist);
    CufftFrontend::AddHostPointerForArguments<int>(onembed);
    CufftFrontend::AddVariableForArguments<int>(ostride);
    CufftFrontend::AddVariableForArguments<int>(odist);
    
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    
    CufftFrontend::Execute("cufftGetSizeMany");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

#if CUDART_VERSION >= 7000
extern "C" cufftResult cufftGetSizeMany64(cufftHandle plan, int rank, long long int *n, 
        long long int *inembed, long long int istride, long long int idist, long long int *onembed, long long int ostride,
        long long int odist, cufftType type, long long int batch,size_t * workSize) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(rank);
    CufftFrontend::AddHostPointerForArguments<long long int>(n);
    CufftFrontend::AddHostPointerForArguments<long long int>(inembed);
    CufftFrontend::AddVariableForArguments<long long int>(istride);
    CufftFrontend::AddVariableForArguments<long long int>(idist);
    CufftFrontend::AddHostPointerForArguments<long long int>(onembed);
    CufftFrontend::AddVariableForArguments<long long int>(ostride);
    CufftFrontend::AddVariableForArguments<long long int>(odist);
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<long long int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    CufftFrontend::Execute("cufftGetSizeMany64");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}
#endif

extern "C" cufftResult cufftGetSize(cufftHandle handle, size_t * workSize){
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(handle);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    CufftFrontend::Execute("cufftGetSize");
    if(CufftFrontend::Success())
        *workSize = *CufftFrontend::GetOutputHostPointer<size_t>();
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, 
        int *inembed, int istride, int idist, int *onembed, int ostride,
        int odist, cufftType type, int batch) {
    CufftFrontend::Prepare();
    //Passing arguments
    CufftFrontend::AddHostPointerForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(rank);
    CufftFrontend::AddHostPointerForArguments<int>(n);
    CufftFrontend::AddHostPointerForArguments<int>(inembed);
    CufftFrontend::AddVariableForArguments<int>(istride);
    CufftFrontend::AddVariableForArguments<int>(idist);
    CufftFrontend::AddHostPointerForArguments<int>(onembed);
    CufftFrontend::AddVariableForArguments<int>(ostride);
    CufftFrontend::AddVariableForArguments<int>(odist);
    
    CufftFrontend::AddVariableForArguments<cufftType>(type);
    CufftFrontend::AddVariableForArguments<int>(batch);
    CufftFrontend::Execute("cufftPlanMany");
    
    if(CufftFrontend::Success())
        plan = CufftFrontend::GetOutputHostPointer<cufftHandle>();
        
    return (cufftResult) CufftFrontend::GetExitCode();
}


extern "C" cufftResult cufftCreate(cufftHandle *plan) {
    CufftFrontend::Prepare();
    CufftFrontend::AddHostPointerForArguments(plan);
    CufftFrontend::Execute("cufftCreate");
    if(CufftFrontend::Success())
        *plan = *(CufftFrontend::GetOutputHostPointer<cufftHandle>());
    //printf("plan: %d",*plan);
    return (cufftResult) CufftFrontend::GetExitCode();//(cufftResult) CufftFrontend::GetExitCode();
}


extern "C" cufftResult cufftDestroy(cufftHandle plan) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::Execute("cufftDestroy");
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftSetWorkArea(cufftHandle plan, void *workArea) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddDevicePointerForArguments(workArea);
    CufftFrontend::Execute("cufftSetWorkArea");
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate){
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(autoAllocate);
    CufftFrontend::Execute("cufftSetAutoAllocation");
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata,
        cufftComplex *odata) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddDevicePointerForArguments((void*)idata);
    CufftFrontend::AddDevicePointerForArguments((void*)odata);
    CufftFrontend::Execute("cufftExecR2C");
    if(CufftFrontend::Success()){
        odata = (cufftComplex*)(CufftFrontend::GetOutputDevicePointer());
    }
    return (cufftResult) CufftFrontend::GetExitCode();
}


extern "C" cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata,
        cufftReal *odata) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddDevicePointerForArguments((void*)idata);
    CufftFrontend::AddDevicePointerForArguments((void*)odata);
    CufftFrontend::Execute("cufftExecC2R");
    if(CufftFrontend::Success()){
        odata = (cufftReal*)(CufftFrontend::GetOutputDevicePointer());
    }
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata,
        cufftDoubleComplex *odata, int direction) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddDevicePointerForArguments((void*)idata);
    CufftFrontend::AddDevicePointerForArguments((void*)odata);
    CufftFrontend::AddVariableForArguments<int>(direction);
    CufftFrontend::Execute("cufftExecZ2Z");
    if(CufftFrontend::Success()){
        odata = (cufftDoubleComplex*)(CufftFrontend::GetOutputDevicePointer());
    }
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata,
        cufftDoubleComplex *odata) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    //cout << "idata: "<< idata << "odata: "<<odata<<endl;
    CufftFrontend::AddDevicePointerForArguments((void*)idata);
    CufftFrontend::AddDevicePointerForArguments((void*)odata);
    CufftFrontend::Execute("cufftExecD2Z");
    if(CufftFrontend::Success())
        odata = (cufftDoubleComplex*)(CufftFrontend::GetOutputDevicePointer());
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata,
        cufftDoubleReal *odata) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddDevicePointerForArguments((void*)idata);
    CufftFrontend::AddDevicePointerForArguments((void*)odata);
    //cout << "idata: "<< idata << "odata: "<<odata<<endl;
    CufftFrontend::Execute("cufftExecZ2D");
    if(CufftFrontend::Success())
        odata = (cufftDoubleReal*)(CufftFrontend::GetOutputDevicePointer());
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) {
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddDevicePointerForArguments(stream);
    CufftFrontend::Execute("cufftSetStream");
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftSetCompatibilityMode(cufftHandle plan,
        cufftCompatibility mode){
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<cufftCompatibility>(mode);
    CufftFrontend::Execute("cufftSetCompatibilityMode");
    return (cufftResult) CufftFrontend::GetExitCode();
}

#if __CUDA_API_VERSION >= 7000
extern "C" cufftResult cufftGetProperty(libraryPropertyType type, int* value){
    CufftFrontend::Prepare();
    CufftFrontend::AddVariableForArguments<libraryPropertyType>(type);
    CufftFrontend::AddHostPointerForArguments<int>(value);
    CufftFrontend::Execute("cufftGetProperty");
    return (cufftResult) CufftFrontend::GetExitCode();
}
#endif

#if __CUDA_API_VERSION >= 7000
extern "C" cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int *onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t *workSize, cudaDataType executiontype) {
    CufftFrontend::Prepare();
    //Passing Arguments
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(rank);
    CufftFrontend::AddHostPointerForArguments<long long int>(n);
    
    CufftFrontend::AddHostPointerForArguments<long long int>(inembed);
    CufftFrontend::AddVariableForArguments<long long int>(istride);
    CufftFrontend::AddVariableForArguments<long long int>(idist);
    CufftFrontend::AddVariableForArguments<cudaDataType>(inputtype);
    
    CufftFrontend::AddHostPointerForArguments<long long int>(onembed);
    CufftFrontend::AddVariableForArguments<long long int>(ostride);
    CufftFrontend::AddVariableForArguments<long long int>(odist);
    CufftFrontend::AddVariableForArguments<cudaDataType>(outputtype);
    
    CufftFrontend::AddVariableForArguments<long long int>(batch);
    CufftFrontend::AddHostPointerForArguments<size_t>(workSize);
    CufftFrontend::AddVariableForArguments<cudaDataType>(executiontype);
    
    CufftFrontend::Execute("cufftXtMakePlanMany");
    if(CufftFrontend::Success()){
        *workSize = *(CufftFrontend::GetOutputHostPointer<size_t>());
    }
    return (cufftResult) CufftFrontend::GetExitCode();
}
#endif

extern "C" cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) {
    CufftFrontend::Prepare();
    //Passing Arguments
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddDevicePointerForArguments((void*)idata);
    CufftFrontend::AddDevicePointerForArguments((void*)odata);
    CufftFrontend::AddVariableForArguments<int>(direction);
    CufftFrontend::Execute("cufftExecC2C");
    if(CufftFrontend::Success()){
        odata = (cufftComplex*)(CufftFrontend::GetOutputDevicePointer());
    }
    return (cufftResult) CufftFrontend::GetExitCode();
}


extern "C" cufftResult cufftXtSetGPUs(cufftHandle plan, int nGPUs, int *whichGPUs) {
    CufftFrontend::Prepare();
    //Passing arguments
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddVariableForArguments<int>(nGPUs);
    CufftFrontend::AddHostPointerForArguments<int>(whichGPUs,sizeof(int));
 
    CufftFrontend::Execute("cufftXtSetGPUs");
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftXtMalloc(cufftHandle plan, cudaLibXtDesc **descriptor,cufftXtSubFormat format ){
    CufftFrontend::Prepare();
    //Passing arguments
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddHostPointerForArguments<cudaLibXtDesc*>(descriptor);
    CufftFrontend::AddVariableForArguments<cufftXtSubFormat>(format);
    
    CufftFrontend::Execute("cufftXtMalloc");
    if (CufftFrontend::Success())
        descriptor = CufftFrontend::GetOutputHostPointer<cudaLibXtDesc*>();
    return (cufftResult) CufftFrontend::GetExitCode();
}


/*Da testare*/
extern "C" cufftResult cufftXtMemcpy(cufftHandle plan, void *dstPointer, void *srcPointer,cufftXtCopyType type){
    CufftFrontend::Prepare();
    
    switch(type){
        case CUFFT_COPY_HOST_TO_DEVICE:
            CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
            CufftFrontend::AddDevicePointerForArguments((void*)dstPointer);
            CufftFrontend::AddHostPointerForArguments((void*)srcPointer);
            CufftFrontend::AddVariableForArguments<cufftXtCopyType>(CUFFT_COPY_HOST_TO_DEVICE);
            CufftFrontend::Execute("cufftXtMemcpy");
            cout<<"executed"<<endl;
            //if(CufftFrontend::Success())
            //    dstPointer = CufftFrontend::GetOutputDevicePointer();
            cout <<"dstPointer"<<dstPointer<<endl;
            break;
        case CUFFT_COPY_DEVICE_TO_HOST:
            CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
            CufftFrontend::AddHostPointerForArguments(dstPointer);
            CufftFrontend::AddDevicePointerForArguments(srcPointer);
            CufftFrontend::AddVariableForArguments<cufftXtCopyType>(type);
            CufftFrontend::Execute("cufftXtMemcpy");
            if(CufftFrontend::Success())
                dstPointer = CufftFrontend::GetOutputHostPointer<void>();
            break;
        case CUFFT_COPY_DEVICE_TO_DEVICE:
            CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
            CufftFrontend::AddDevicePointerForArguments(dstPointer);
            CufftFrontend::AddDevicePointerForArguments(srcPointer);
            CufftFrontend::AddVariableForArguments<cufftXtCopyType>(type);
            CufftFrontend::Execute("cufftXtMemcpy");
            if(CufftFrontend::Success())
                dstPointer = CufftFrontend::GetOutputDevicePointer();
            break;
        default:
            break;
    }
    return (cufftResult) CufftFrontend::GetExitCode();
}


/*Da testare*/
extern "C" cufftResult cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc *input, cudaLibXtDesc *output, int direction){
    CufftFrontend::Prepare();
    //Passing Arguments
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddDevicePointerForArguments(input);
    CufftFrontend::AddDevicePointerForArguments(output);
    CufftFrontend::AddVariableForArguments<int>(direction);
    
    CufftFrontend::Execute("cufftXtExecDescriptorC2C");
    if(CufftFrontend::Success()){
        output = (cudaLibXtDesc*)(CufftFrontend::GetOutputDevicePointer());
    }
    return (cufftResult) CufftFrontend::GetExitCode();
}


/*Da testare*/
extern "C" cufftResult cufftXtFree(cudaLibXtDesc *descriptor){
    CufftFrontend::Prepare();
    //Passing arguments
    CufftFrontend::AddHostPointerForArguments<cudaLibXtDesc>(descriptor);
    CufftFrontend::Execute("cufftXtFree");
    return (cufftResult) CufftFrontend::GetExitCode();
}

/*Da testare*/
extern "C" cufftResult cufftXtSetCallback(cufftHandle plan, void **callbackRoutine, cufftXtCallbackType type, void **callerInfo){
    /*CufftFrontend::Prepare();
    //Passing arguments
    CufftFrontend::AddVariableForArguments<cufftHandle>(plan);
    CufftFrontend::AddHostPointerForArguments(callbackRoutine);
    CufftFrontend::AddVariableForArguments<cufftXtCallbackType>(type);
    CufftFrontend::AddDevicePointerForArguments(callerInfo);
    cout<<"callerinfo: "<<callerInfo<< "plan: "<<plan<<endl;
    CufftFrontend::Execute("cufftXtSetCallback");*/
    /* Avoiding useless communication because GVIRTUS does not support statically linked libraries */
    cout << endl <<"EXCEPTION - function cufftXtSetCallback not supported in GVIRTUS" << endl;
    return (cufftResult) CUFFT_NOT_IMPLEMENTED;//CufftFrontend::GetExitCode();
} 

extern "C" cufftResult cufftGetVersion(int *version){
    CufftFrontend::Prepare();
    CufftFrontend::AddHostPointerForArguments<int>(version);
    CufftFrontend::Execute("cufftGetVersion");
    return (cufftResult) CufftFrontend::GetExitCode();
}

