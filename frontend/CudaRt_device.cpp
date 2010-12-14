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

extern "C" cudaError_t cudaChooseDevice(int *device, const cudaDeviceProp *prop) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(device);
    f->AddHostPointerForArguments(prop);
    f->Execute("cudaChooseDevice");
    if(f->Success())
        *device = *(f->GetOutputHostPointer<int>());
    return f->GetExitCode();
}

extern "C" cudaError_t cudaGetDevice(int *device) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(device);
    f->Execute("cudaGetDevice");
    if(f->Success())
        *device = *(f->GetOutputHostPointer<int>());
    return f->GetExitCode();
}

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(count);
    f->Execute("cudaGetDeviceCount");
    if(f->Success())
        *count = *(f->GetOutputHostPointer<int>());
    return f->GetExitCode();
}

extern "C" cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(prop);
    f->AddVariableForArguments(device);
    f->Execute("cudaGetDeviceProperties");
    if(f->Success()) {
        memmove(prop, f->GetOutputHostPointer<cudaDeviceProp>(),
                sizeof(cudaDeviceProp));
#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION >= 2030 
        prop->canMapHostMemory = 0;
#endif
    }
    return f->GetExitCode();
}

extern "C" cudaError_t cudaSetDevice(int device) {
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(device);
    f->Execute("cudaSetDevice");
    return f->GetExitCode();
}

#if CUDART_VERSION >= 3000
extern "C" cudaError_t cudaSetDeviceFlags(unsigned int flags) {
#else
extern "C" cudaError_t cudaSetDeviceFlags(int flags) {
#endif
    Frontend *f = Frontend::GetFrontend();
    f->AddVariableForArguments(flags);
    f->Execute("cudaSetDeviceFlags");
    return f->GetExitCode();
}

extern "C" cudaError_t cudaSetValidDevices(int *device_arr, int len) {
    Frontend *f = Frontend::GetFrontend();
    f->AddHostPointerForArguments(device_arr, len);
    f->AddVariableForArguments(len);
    f->Execute("cudaSetValidDevices");
    if(f->Success()) {
        int *out_device_arr = f->GetOutputHostPointer<int>();
        memmove(device_arr, out_device_arr, sizeof(int) * len);
    }
    return f->GetExitCode();
}
