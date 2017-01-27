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

#include <cufft.h>

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
    cout << "plan : "<< *plan;
    
    return (cufftResult) CufftFrontend::GetExitCode();
}

extern "C" cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny,
        cufftType type) {
    CufftFrontend::Prepare();
    CufftFrontend::AddHostPointerForArguments<cufftHandle>(plan);
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
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(nx);
    in->Add(ny);
    in->Add(nz);
    in->Add(type);
    f->Execute("cufftPlan3d");
    *plan = (cufftHandle) f->GetOutputBuffer()->Get<uint64_t>();
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n, 
        int *inembed, int istride, int idist, int *onembed, int ostride,
        int odist, cufftType type, int batch) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(rank);
    in->Add(*n);
    in->Add(istride);
    in->Add(idist);
    in->Add(ostride);
    in->Add(odist);
    in->Add(type);
    in->Add(batch);
    f->Execute("cufftPlanMany");
    *plan = (cufftHandle) f->GetOutputBuffer()->Get<uint64_t>();
    *n = f->GetOutputBuffer()->Get<int>();
    return (cufftResult) f->GetExitCode();
}

/*
 * in Testing - Vincenzo Santopietro
 */
extern "C" cufftResult cufftCreate(cufftHandle *plan) {
    //Frontend *f = Frontend::GetFrontend();
    CufftFrontend::Prepare();
    CufftFrontend::AddHostPointerForArguments(plan);
    //f->GetFrontend();
    CufftFrontend::Execute("cufftCreate");
    if(CufftFrontend::Success())
        *plan = *(CufftFrontend::GetOutputHostPointer<cufftHandle>());
    printf("plan: %d",*plan);
    return (cufftResult) CufftFrontend::GetExitCode();//(cufftResult) CufftFrontend::GetExitCode();
}


extern "C" cufftResult cufftDestroy(cufftHandle plan) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    f->Execute("cufftDestroy");
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftExecC2C(cufftHandle plan, cufftComplex *idata,
        cufftComplex *odata, int direction) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    in->Add(idata);
    in->Add(odata);
    in->Add(direction);
    f->Execute("cufftExecC2C");
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftExecR2C(cufftHandle plan, cufftReal *idata,
        cufftComplex *odata) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    in->Add(idata);
    in->Add(odata);
    f->Execute("cufftExecR2C");
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata,
        cufftReal *odata) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    in->Add((uint64_t) idata);
    in->Add((uint64_t) odata);
    f->Execute("cufftExecC2R");
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata,
        cufftDoubleComplex *odata, int direction) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    in->Add(idata);
    in->Add(odata);
    in->Add(direction);
    f->Execute("cufftExecZ2z");
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal *idata,
        cufftDoubleComplex *odata) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    in->Add(idata);
    in->Add(odata);
    f->Execute("cufftExecD2Z");
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex *idata,
        cufftDoubleReal *odata) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    in->Add(idata);
    in->Add(odata);
    f->Execute("cufftExecZ2D");
    return (cufftResult) f->GetExitCode();

}

extern "C" cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    in->Add((uint64_t) stream);
    f->Execute("cufftSetStream");
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftSetCompatibilityMode(cufftHandle plan,
        cufftCompatibility mode){
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(plan);
    in->Add(mode);
    f->Execute("cufftSetCompatibilityMode");
    return (cufftResult) f->GetExitCode();
}


extern "C" cufftResult cufftXtMakePlanMany(cufftHandle plan, int rank, long long int *n, long long int *inembed, long long int istride, long long int idist, cudaDataType inputtype, long long int *onembed, long long int ostride, long long int odist, cudaDataType outputtype, long long int batch, size_t *workSize, cudaDataType executiontype) {
    /*Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *input_buffer = f->GetInputBuffer();
    plan = (cufftHandle) f->GetOutputBuffer()->Get<cufftHandle>();
    n = (long long int * )f->GetOutputBuffer()->Get<long long int>();
    
    input_buffer->Add(plan);
    input_buffer->Add(rank);
    input_buffer->Add(*n);
    input_buffer->Add(*inembed);
    input_buffer->Add(istride);
    input_buffer->Add(idist);
    input_buffer->Add(inputtype);

    input_buffer->Add(*onembed);
    input_buffer->Add(ostride);
    input_buffer->Add(odist);
    input_buffer->Add(outputtype);
    
    input_buffer->Add(batch);
    input_buffer->Add(*workSize);
    input_buffer->Add(executiontype);
    f->Execute("cufftXtMakePlanMany");
    
    return (cufftResult) f->GetExitCode();*/
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
        //*n = *(CufftFrontend::GetOutputHostPointer<long long int>());
        //*inembed = *(CufftFrontend::GetOutputHostPointer<long long int>());
        //*onembed = *(CufftFrontend::GetOutputHostPointer<long long int>());
        *workSize = *(CufftFrontend::GetOutputHostPointer<size_t>());
    }
    return (cufftResult) CufftFrontend::GetExitCode();
}
