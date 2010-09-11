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

#include <cuda_runtime_api.h>
#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(ChooseDevice) {
    int *device = input_buffer->Assign<int>();
    const cudaDeviceProp *prop = input_buffer->Assign<cudaDeviceProp>();

    cudaError_t exit_code = cudaChooseDevice(device, prop);

    Buffer *out = new Buffer();
    out->Add(device);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDevice) {
    int *device = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaGetDevice(device);

    Buffer *out = new Buffer();
    out->Add(device);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDeviceCount) {
    int *count = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaGetDeviceCount(count);

    Buffer *out = new Buffer();
    out->Add(count);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDeviceProperties) {
    struct cudaDeviceProp *prop = input_buffer->Assign<struct cudaDeviceProp>();
    int device = input_buffer->Get<int>();

    cudaError_t exit_code = cudaGetDeviceProperties(prop, device);
#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
    prop->canMapHostMemory = 0;
#endif

    Buffer *out = new Buffer();
    out->Add(prop);

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(SetDevice) {
    int device = input_buffer->Get<int>();

    cudaError_t exit_code = cudaSetDevice(device);

    return new Result(exit_code);
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
CUDA_ROUTINE_HANDLER(SetDeviceFlags) {
    int flags = input_buffer->Get<int>();

    cudaError_t exit_code = cudaSetDeviceFlags(flags);

    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(SetValidDevices) {
    int len = input_buffer->BackGet<int>();
    int *device_arr = input_buffer->Assign<int>(len);

    cudaError_t exit_code = cudaSetValidDevices(device_arr, len);

    Buffer *out = new Buffer();
    out->Add(device_arr, len);

    return new Result(exit_code, out);
}
#endif

