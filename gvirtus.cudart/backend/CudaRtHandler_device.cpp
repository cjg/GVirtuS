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

#include "CudaRtHandler.h"

CUDA_ROUTINE_HANDLER(DeviceSetCacheConfig) {
    try {
        cudaFuncCache cacheConfig = input_buffer->Get<cudaFuncCache>();
        cudaError_t exit_code = cudaDeviceSetCacheConfig(cacheConfig);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation); //???
    }
}

CUDA_ROUTINE_HANDLER(DeviceSetLimit) {
    try {
        cudaLimit limit = input_buffer->Get<cudaLimit>();
        size_t value = input_buffer->Get<size_t>();
        cudaError_t exit_code = cudaDeviceSetLimit(limit, value);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation); //???
    }
}

CUDA_ROUTINE_HANDLER(IpcOpenMemHandle) {
    void *devPtr = NULL;
    try {
        cudaIpcMemHandle_t handle = input_buffer->Get<cudaIpcMemHandle_t>();
        unsigned int flags = input_buffer->Get<unsigned int>();
        cudaError_t exit_code = cudaIpcOpenMemHandle(&devPtr, handle, flags);
        Buffer *out = new Buffer();
        out->AddMarshal(devPtr);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(DeviceEnablePeerAccess) {
    int peerDevice = input_buffer->Get<int>();
    unsigned int flags = input_buffer->Get<unsigned int>();
    cudaError_t exit_code = cudaDeviceEnablePeerAccess(peerDevice, flags);
    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(DeviceDisablePeerAccess) {
    int peerDevice = input_buffer->Get<int>();
    cudaError_t exit_code = cudaDeviceDisablePeerAccess(peerDevice);
    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(DeviceCanAccessPeer) {
    int *canAccessPeer = input_buffer->Assign<int>();
    int device = input_buffer->Get<int>();
    int peerDevice = input_buffer->Get<int>();

    cudaError_t exit_code = cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);

    Buffer *out = new Buffer();
    try {
        out->Add(canAccessPeer);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceGetStreamPriorityRange) {
    int *leastPriority = input_buffer->Assign<int>();
    int *greatestPriority = input_buffer->Assign<int>();

    cudaError_t exit_code = cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority);

    Buffer *out = new Buffer();
    try {
        out->Add(leastPriority);
        out->Add(greatestPriority);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(OccupancyMaxActiveBlocksPerMultiprocessor) {
    int *numBlocks = input_buffer->Assign<int>();
    const char *func = (const char*) (input_buffer->Get<pointer_t> ());
    int blockSize = input_buffer->Get<int>();
    size_t dynamicSMemSize = input_buffer->Get<size_t>();

    cudaError_t exit_code = cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynamicSMemSize);

    Buffer *out = new Buffer();
    try {
        out->Add(numBlocks);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceGetAttribute) {
    int *value = input_buffer->Assign<int>();
    cudaDeviceAttr attr = input_buffer->Get<cudaDeviceAttr>();
    int device = input_buffer->Get<int>();
    cudaError_t exit_code = cudaDeviceGetAttribute(value, attr, device);
    Buffer *out = new Buffer();
    try {
        out->Add(value);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(IpcGetMemHandle) {
    cudaIpcMemHandle_t *handle = input_buffer->Assign<cudaIpcMemHandle_t>();
    void *devPtr = input_buffer->GetFromMarshal<void *>();

    cudaError_t exit_code = cudaIpcGetMemHandle(handle, devPtr);

    Buffer *out = new Buffer();
    try {
        out->Add(handle);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(IpcGetEventHandle) {

    cudaIpcEventHandle_t *handle = input_buffer->Assign<cudaIpcEventHandle_t>();
    cudaEvent_t event = input_buffer->Get<cudaEvent_t>();

    cudaError_t exit_code = cudaIpcGetEventHandle(handle, event);

    Buffer *out = new Buffer();
    try {
        out->Add(handle);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(ChooseDevice) {
    int *device = input_buffer->Assign<int>();
    const cudaDeviceProp *prop = input_buffer->Assign<cudaDeviceProp>();
    cudaError_t exit_code = cudaChooseDevice(device, prop);
    Buffer *out = new Buffer();
    try {
        out->Add(device);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(GetDevice) {
    try {
        int *device = input_buffer->Assign<int>();
        cudaError_t exit_code = cudaGetDevice(device);
        Buffer *out = new Buffer();
        out->Add(device);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(DeviceReset) {
    cudaError_t exit_code = cudaDeviceReset();
    Buffer *out = new Buffer();
    return new Result(exit_code, out);
}

CUDA_ROUTINE_HANDLER(DeviceSynchronize) {

    cudaError_t exit_code = cudaDeviceSynchronize();

    try {
        Buffer *out = new Buffer();
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(GetDeviceCount) {
    try {
        int *count = input_buffer->Assign<int>();
        cudaError_t exit_code = cudaGetDeviceCount(count);
        Buffer *out = new Buffer();
        out->Add(count);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(GetDeviceProperties) {
    try {
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
        out->Add(prop, 1);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(SetDevice) {
    try {
        int device = input_buffer->Get<int>();
        cudaError_t exit_code = cudaSetDevice(device);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030

CUDA_ROUTINE_HANDLER(SetDeviceFlags) {
    try {
        int flags = input_buffer->Get<int>();
        cudaError_t exit_code = cudaSetDeviceFlags(flags);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(IpcOpenEventHandle) {
    Buffer *out = new Buffer();
    try {
        cudaEvent_t *event = input_buffer->Assign<cudaEvent_t>();
        cudaIpcEventHandle_t handle = input_buffer->Get<cudaIpcEventHandle_t>();
        cudaError_t exit_code = cudaIpcOpenEventHandle(event, handle);
        out->Add(event);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
}

CUDA_ROUTINE_HANDLER(SetValidDevices) {
    try {
        int len = input_buffer->BackGet<int>();
        int *device_arr = input_buffer->Assign<int>(len);
        cudaError_t exit_code = cudaSetValidDevices(device_arr, len);
        Buffer *out = new Buffer();
        out->Add(device_arr, len);
        return new Result(exit_code, out);

    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}
#endif

