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

#include "Frontend.h"
#include "CudaRt.h"

using namespace std;

extern cudaError_t cudaEventCreate(cudaEvent_t *event) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(event);
    f->Execute("cudaEventCreate");
    if(f->Success())
        *event = *(f->GetOutputHostPointer<cudaEvent_t>());
    return f->GetExitCode();
}

extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, int flags) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(event);
    f->AddVariableForArguments(flags);
    f->Execute("cudaEventCreateWithFlags");
    if(f->Success())
        *event = *(f->GetOutputHostPointer<cudaEvent_t>());
    return f->GetExitCode();
}

extern cudaError_t cudaEventDestroy(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(event);
    f->Execute("cudaEventDestroy");
    return f->GetExitCode();
}

extern cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(ms);
    f->AddVariableForArguments(start);
    f->AddVariableForArguments(end);
    f->Execute("cudaEventElapsedTime");
    if(f->Success())
        *ms = *(f->GetOutputHostPointer<float>());
    return f->GetExitCode();
}

extern cudaError_t cudaEventQuery(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(event);
    f->Execute("cudaEventQuery");
    return f->GetExitCode();
}

extern cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(event);
    f->AddVariableForArguments(stream);
    f->Execute("cudaEventRecord");
    return f->GetExitCode();
}

extern cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(event);
    f->Execute("cudaEventSynchronize");
    return f->GetExitCode();
}
