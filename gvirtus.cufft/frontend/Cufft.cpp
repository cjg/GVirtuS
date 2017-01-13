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

#include <iostream>

using namespace std;

extern "C" cufftResult cufftPlan1d(cufftHandle *plan, int nx, cufftType type,
        int batch) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(nx);
    in->Add(type);
    in->Add(batch);
    f->Execute("cufftPlan1d");
    *plan = (cufftHandle) f->GetOutputBuffer()->Get<uint64_t>();
    return (cufftResult) f->GetExitCode();
}

extern "C" cufftResult cufftPlan2d(cufftHandle *plan, int nx, int ny,
        cufftType type) {
    Frontend *f = Frontend::GetFrontend();
    f->Prepare();
    Buffer *in = f->GetInputBuffer();
    in->Add(nx);
    in->Add(ny);
    in->Add(type);
    f->Execute("cufftPlan2d");
    *plan = f->GetOutputBuffer()->Get<cufftHandle>();
    return (cufftResult) f->GetExitCode();
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