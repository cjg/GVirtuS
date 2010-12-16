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

#include "CudaRt.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif

using namespace std;

extern "C" __host__ cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->Execute("cudaEventCreate");
    if(f->Success())
        *event = (cudaEvent_t) f->GetOutputDevicePointer();
#else
    f->AddHostPointerForArguments(event);
    f->Execute("cudaEventCreate");
    if(f->Success())
        *event = *(f->GetOutputHostPointer<cudaEvent_t>());
#endif
    return f->GetExitCode();
}

#if CUDART_VERSION >= 3000
extern "C" __host__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event,
        unsigned int flags) {
#else
extern "C" __host__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
#endif
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->AddVariableForArguments(flags);
    f->Execute("cudaEventCreateWithFlags");
    if(f->Success())
        *event = (cudaEvent_t) f->GetOutputDevicePointer();
#else
    f->AddHostPointerForArguments(event);
    f->AddVariableForArguments(flags);
    f->Execute("cudaEventCreateWithFlags");
    if(f->Success())
        *event = *(f->GetOutputHostPointer<cudaEvent_t>());
#endif
    return f->GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->AddDevicePointerForArguments(event);
#else
    f->AddVariableForArguments(event);
#endif
    f->Execute("cudaEventDestroy");
    return f->GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(ms);
#if CUDART_VERSION >= 3010
    f->AddDevicePointerForArguments(start);
    f->AddDevicePointerForArguments(end);
#else
    f->AddVariableForArguments(start);
    f->AddVariableForArguments(end);
#endif
    f->Execute("cudaEventElapsedTime");
    if(f->Success())
        *ms = *(f->GetOutputHostPointer<float>());
    return f->GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->AddDevicePointerForArguments(event);
#else
    f->AddVariableForArguments(event);
#endif
    f->Execute("cudaEventQuery");
    return f->GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->AddDevicePointerForArguments(event);
    f->AddDevicePointerForArguments(stream);
#else
    f->AddVariableForArguments(event);
    f->AddVariableForArguments(stream);
#endif
    f->Execute("cudaEventRecord");
    return f->GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->AddDevicePointerForArguments(event);
#else
    f->AddVariableForArguments(event);
#endif    
    f->Execute("cudaEventSynchronize");
    return f->GetExitCode();
}
