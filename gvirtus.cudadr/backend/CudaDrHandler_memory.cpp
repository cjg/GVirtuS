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
 * Written by: Flora Giannone <flora.giannone@studenti.uniparthenope.it>,
 *             Department of Applied Science
 */

#include <cuda.h>
#include "CudaUtil.h"
#include "CudaDrHandler.h"
#include <stdio.h>
#include <vector>
#include "CudaDrBackend.h"
using namespace std;

/*Frees device memory.*/
CUDA_DRIVER_HANDLER(MemFree) {
    CUdeviceptr dptr = input_buffer->Get<CUdeviceptr > ();
    CUresult exit_code = cuMemFree(dptr);
    return new Result((cudaError_t) exit_code);
}

/*Allocates device memory.*/
CUDA_DRIVER_HANDLER(MemAlloc) {
    CUdeviceptr dptr = 0;
    size_t bytesize = input_buffer->Get<size_t > ();
    CUresult exit_code = cuMemAlloc(&dptr, bytesize);
    Buffer *out = new Buffer();
    out->AddMarshal(dptr);
    return new Result((cudaError_t) exit_code, out);
}

/*Copies memory from Device to Host. */
CUDA_DRIVER_HANDLER(MemcpyDtoH) {
    CUdeviceptr srcDevice = input_buffer->Get<CUdeviceptr > ();
    size_t ByteCount = input_buffer->Get<size_t > ();
    void *dstHost = new char[ByteCount];
    CUresult exit_code = cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
    Buffer * out = new Buffer();
    out->Add<char>((char *) dstHost, ByteCount);
    delete[] (char *) dstHost;
    return new Result((cudaError_t) exit_code, out);
}

/*Copies memory from Host to Device.*/
CUDA_DRIVER_HANDLER(MemcpyHtoD) {
    void *srcHost = NULL;
    size_t ByteCount = input_buffer->Get<size_t > ();
    CUdeviceptr dstDevice = input_buffer->Get<CUdeviceptr > ();
    srcHost = input_buffer->Assign<char>(ByteCount);
    CUresult exit_code = cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
    return new Result((cudaError_t) exit_code);
}

/*Creates a 1D or 2D CUDA array. */
CUDA_DRIVER_HANDLER(ArrayCreate) {
    CUarray pHandle;
    const CUDA_ARRAY_DESCRIPTOR *pAllocateArray = input_buffer->Assign<const CUDA_ARRAY_DESCRIPTOR > ();
    CUresult exit_code = cuArrayCreate(&pHandle, pAllocateArray);
    Buffer *out = new Buffer();
    out->AddMarshal(pHandle);
    return new Result((cudaError_t) exit_code, out);
}

/*Creates a 3D CUDA array.*/
CUDA_DRIVER_HANDLER(Array3DCreate) {
    CUarray pHandle;
    const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray = input_buffer->Assign<const CUDA_ARRAY3D_DESCRIPTOR > ();
    CUresult exit_code = cuArray3DCreate(&pHandle, pAllocateArray);
    Buffer *out = new Buffer();
    out->AddMarshal(pHandle);
    return new Result((cudaError_t) exit_code, out);
}

/*Copies memory for 2D arrays. */
CUDA_DRIVER_HANDLER(Memcpy2D) {
    CUDA_MEMCPY2D *pCopy = input_buffer->AssignAll< CUDA_MEMCPY2D > ();
    int flag = input_buffer->Get<int>();
    if (flag == 1)
        pCopy->srcHost = input_buffer->AssignAll<char>();
    CUresult exit_code = cuMemcpy2D(pCopy);
    return new Result((cudaError_t) exit_code);
}

/*Destroys a CUDA array.*/
CUDA_DRIVER_HANDLER(ArrayDestroy) {
    CUarray hArray = input_buffer->Get<CUarray > ();
    CUresult exit_code = cuArrayDestroy(hArray);
    return new Result((cudaError_t) exit_code);
}

/*Allocates pitched device memory.*/
CUDA_DRIVER_HANDLER(MemAllocPitch) {
    CUdeviceptr dptr;
    size_t pitch;
    size_t WidthInBytes = input_buffer->Get<size_t > ();
    size_t Height = input_buffer->Get<size_t > ();
    unsigned int ElementSizeBytes = input_buffer->Get<unsigned int>();
    CUresult exit_code = cuMemAllocPitch(&dptr, &pitch, WidthInBytes, Height, ElementSizeBytes);
    Buffer *out = new Buffer();
    out->AddMarshal(dptr);
    out->Add(pitch);
    return new Result((cudaError_t) exit_code, out);
}

/*Get information on memory allocations.*/
CUDA_DRIVER_HANDLER(MemGetAddressRange) {
    CUdeviceptr pbase;
    size_t psize;
    CUdeviceptr dptr = input_buffer->Get<CUdeviceptr > ();
    CUresult exit_code = cuMemGetAddressRange(&pbase, &psize, dptr);
    Buffer *out = new Buffer();
    out->AddMarshal(pbase);
    out->Add(psize);
    return new Result((cudaError_t) exit_code, out);
}

/*Gets free and total memory.*/
CUDA_DRIVER_HANDLER(MemGetInfo) {
    size_t free;
    size_t total;
    CUresult exit_code = cuMemGetInfo(&free, &total);
    Buffer *out = new Buffer();
    out->Add(free);
    out->Add(total);
    return new Result((cudaError_t) exit_code, out);
}
