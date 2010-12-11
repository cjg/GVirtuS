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

using namespace std;

extern "C" cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->Execute("cudaStreamCreate");
    if(f->Success())
        *pStream = (cudaStream_t) f->GetOutputDevicePointer();
#else
    f->AddHostPointerForArguments(pStream);
    f->Execute("cudaStreamCreate");
    if(f->Success())
        *pStream = *(f->GetOutputHostPointer<cudaStream_t>());
#endif
    return f->GetExitCode();
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->AddDevicePointerForArguments(stream);
#else
    f->AddVariableForArguments(stream);
#endif    
    f->Execute("cudaStreamDestroy");
    return f->GetExitCode();
}

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->AddDevicePointerForArguments(stream);
#else
    f->AddVariableForArguments(stream);
#endif     
    f->Execute("cudaStreamQuery");
    return f->GetExitCode();
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    Frontend *f = Frontend::GetFrontend();
#if CUDART_VERSION >= 3010
    f->AddDevicePointerForArguments(stream);
#else
    f->AddVariableForArguments(stream);
#endif     
    f->Execute("cudaStreamSynchronize");
    return f->GetExitCode();
}
