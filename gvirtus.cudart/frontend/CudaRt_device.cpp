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

extern "C" __host__ cudaError_t CUDARTAPI cudaChooseDevice(int *device, const cudaDeviceProp *prop) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(device);
    CudaRtFrontend::AddHostPointerForArguments(prop);
    CudaRtFrontend::Execute("cudaChooseDevice");
    if(CudaRtFrontend::Success())
        *device = *(CudaRtFrontend::GetOutputHostPointer<int>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetDevice(int *device) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(device);
    CudaRtFrontend::Execute("cudaGetDevice");
    if(CudaRtFrontend::Success())
        *device = *(CudaRtFrontend::GetOutputHostPointer<int>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(count);
    CudaRtFrontend::Execute("cudaGetDeviceCount");
    if(CudaRtFrontend::Success())
        *count = *(CudaRtFrontend::GetOutputHostPointer<int>());
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(prop);
    CudaRtFrontend::AddVariableForArguments(device);
    CudaRtFrontend::Execute("cudaGetDeviceProperties");
    if(CudaRtFrontend::Success()) {
        memmove(prop, CudaRtFrontend::GetOutputHostPointer<cudaDeviceProp>(),
                sizeof(cudaDeviceProp));
#ifndef CUDA_VERSION
#error CUDA_VERSION not defined
#endif
#if CUDA_VERSION >= 2030 
        prop->canMapHostMemory = 0;
#endif
    }
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaSetDevice(int device) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(device);
    CudaRtFrontend::Execute("cudaSetDevice");
    return CudaRtFrontend::GetExitCode();
}

#if CUDART_VERSION >= 3000
extern "C" __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags(unsigned int flags) {
#else
extern "C" __host__ cudaError_t CUDARTAPI cudaSetDeviceFlags(int flags) {
#endif
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddVariableForArguments(flags);
    CudaRtFrontend::Execute("cudaSetDeviceFlags");
    return CudaRtFrontend::GetExitCode();
}

extern "C" __host__ cudaError_t CUDARTAPI cudaDeviceReset(void) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaDeviceReset");
    return CudaRtFrontend::GetExitCode();

}

extern "C" __host__ cudaError_t CUDARTAPI cudaDeviceSynchronize(void) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::Execute("cudaDeviceSynchronize");
    return CudaRtFrontend::GetExitCode();

}

extern "C" __host__ cudaError_t CUDARTAPI cudaSetValidDevices(int *device_arr, int len) {
    CudaRtFrontend::Prepare();
    CudaRtFrontend::AddHostPointerForArguments(device_arr, len);
    CudaRtFrontend::AddVariableForArguments(len);
    CudaRtFrontend::Execute("cudaSetValidDevices");
    if(CudaRtFrontend::Success()) {
        int *out_device_arr = CudaRtFrontend::GetOutputHostPointer<int>();
        memmove(device_arr, out_device_arr, sizeof(int) * len);
    }
    return CudaRtFrontend::GetExitCode();
}
