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

CUDA_ROUTINE_HANDLER(ConfigureCall) {
    /* cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
     * size_t sharedMem, cudaStream_t stream) */
    fprintf(stderr, "cudaConfigureCall\n\n");
    try {
        dim3 gridDim = input_buffer->Get<dim3>();
        dim3 blockDim = input_buffer->Get<dim3>();
        size_t sharedMem = input_buffer->Get<size_t>();
        cudaStream_t stream = input_buffer->Get<cudaStream_t>();
        cudaError_t exit_code = cudaConfigureCall(gridDim, blockDim, sharedMem,stream);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

    //std::cerr << "gridDim: " << gridDim.x << " " << gridDim.y << " " << gridDim.z << " " << std::endl;

    
}

#ifndef CUDART_VERSION
#error CUDART_VERSION not defined
#endif
#if CUDART_VERSION >= 2030
CUDA_ROUTINE_HANDLER(FuncGetAttributes) {
    try {
        cudaFuncAttributes *guestAttr = input_buffer->Assign<cudaFuncAttributes>();
        const char *handler = (const char*)(input_buffer->Get<pointer_t> ());
        Buffer * out = new Buffer();
        cudaFuncAttributes *attr = out->Delegate<cudaFuncAttributes>();
        memmove(attr, guestAttr, sizeof(cudaFuncAttributes));
        cudaError_t exit_code = cudaFuncGetAttributes(attr, handler);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
    

}
#endif

CUDA_ROUTINE_HANDLER(FuncSetCacheConfig) {
    try {
        //(const char*)(input_buffer->Get<pointer_t> ())
        const char *handler = (const char*)(input_buffer->Get<pointer_t> ());
        //const char *entry = pThis->GetDeviceFunction(handler);
        cudaFuncCache cacheConfig = input_buffer->Get<cudaFuncCache>();
        Buffer * out = new Buffer();
        cudaError_t exit_code = cudaFuncSetCacheConfig(handler, cacheConfig);
        return new Result(exit_code, out);
} catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }
    
  
}

CUDA_ROUTINE_HANDLER(Launch) {
    int ctrl;
    void *pointer;
    // cudaConfigureCall
    ctrl = input_buffer->Get<int>();
    if(ctrl != 0x434e34c)
        throw "Expecting cudaConfigureCall";

    dim3 gridDim = input_buffer->Get<dim3>();
    dim3 blockDim = input_buffer->Get<dim3>();
    size_t sharedMem = input_buffer->Get<size_t>();
    cudaStream_t stream = input_buffer->Get<cudaStream_t>();
   
    cudaError_t exit_code = cudaConfigureCall(gridDim, blockDim, sharedMem,
            stream);

    if(exit_code != cudaSuccess)
        return new Result(exit_code);

    // cudaSetupArgument
    
    while((ctrl = input_buffer->Get<int>()) == 0x53544147) {
        void *arg = input_buffer->AssignAll<char>();
        size_t size = input_buffer->Get<size_t>();
        size_t offset = input_buffer->Get<size_t>();
        //fprintf(stderr,"cudaSetupArgument:\n");
        exit_code = cudaSetupArgument(arg, size, offset);
        if(exit_code != cudaSuccess)
            return new Result(exit_code);
    }
   

    // cudaLaunch
    if(ctrl != 0x4c41554e)
        throw "Expecting cudaLaunch";

    
    //char *handler = input_buffer->AssignString();
    //fprintf(stderr,"handler:%s\n",handler); 
    //const char *entry = pThis->GetDeviceFunction(handler);
    
    const char *entry = (const char *)(input_buffer->Get<pointer_t> ());
   
    //fprintf(stderr,"entry:%s\n",entry);
    // //sscanf(entry,"%p",&pointer);
    // //const unsigned long long int* data = (const unsigned long long int*)entry;
    //std::cerr << "cudaConfigureCall executed: " << entry << std::endl;
    // exit_code = cudaLaunch(entry);
    //sscanf(handler,"%p",&pointer);
    //char *__f = ((char *)((void ( *)(const float *, const float *, float *, int))pointer));
    char *__f = ((char *)pointer);
    //fprintf(stderr,"__f:%x\n",__f);

    //entry=(const char *)0x40137b;
    //fprintf(stderr,"Before cuda launch entry_addr:%p\n",entry);
    exit_code = cudaLaunch(entry);
    //fprintf(stderr,"After cuda launch exit code:%d\n", exit_code);
    return new Result(exit_code);
}

CUDA_ROUTINE_HANDLER(SetDoubleForDevice) {
    try {
        double *guestD = input_buffer->Assign<double>();
        Buffer *out = new Buffer();
        double *d = out->Delegate<double>();
        memmove(d, guestD, sizeof(double));
        cudaError_t exit_code = cudaSetDoubleForDevice(d);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(SetDoubleForHost) {
    try {
        double *guestD = input_buffer->Assign<double>();
        Buffer *out = new Buffer();
        double *d = out->Delegate<double>();
        memmove(d, guestD, sizeof(double));
        cudaError_t exit_code = cudaSetDoubleForHost(d);
        return new Result(exit_code, out);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }

}

CUDA_ROUTINE_HANDLER(SetupArgument) {
    /* cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) */
    try {
        size_t offset = input_buffer->BackGet<size_t>();
        size_t size = input_buffer->BackGet<size_t>();
        void *arg = input_buffer->Assign<char>(size);
        cudaError_t exit_code = cudaSetupArgument(arg, size, offset);
        return new Result(exit_code);
    } catch (string e) {
        cerr << e << endl;
        return new Result(cudaErrorMemoryAllocation);
    }


}
